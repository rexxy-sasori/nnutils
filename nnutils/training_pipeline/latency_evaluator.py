"""
All latency measurements are returned in ms
"""
import os
import time

import numpy as np
import scipy.sparse
import sparse_dot_mkl
import torch

from nnutils.training_pipeline.trainers.utils import SuppressPrints


def evaluate_latency(model_state, inputs, target_kernel, target_device, num_trials, verbose=False):
    with SuppressPrints(not verbose):
        latency_ms, layerwise_latency_ms = PLATFORM_LATENCY_EVAL[(target_kernel, target_device)](
            model_state, inputs, target_device, num_trials
        )

    return latency_ms, layerwise_latency_ms


def eval_sparsednn(model_state, inputs_dict, target_device, num_trials):
    total_latency = 0

    print('measuring latency for model')
    time_total = 0

    layerwise_avg_latency = []
    os.chdir('sparsednn')
    for idx, (target_layer_name, input_detail) in enumerate(inputs_dict.items()):
        weight_tensor = model_state[target_layer_name]
        if weight_tensor.dim() == 4:
            cout, cin, hk, wk = weight_tensor.shape
            M = cout
            K = cin * hk * wk
            N = input_detail.output_dim[-1] * input_detail.output_dim[-2]
            weight = weight_tensor.reshape(M, K).to(target_device).numpy()
        else:
            M = weight_tensor.shape[0]
            K = weight_tensor.shape[1]
            N = 1
            weight = weight_tensor.to(target_device).numpy()

        print('start measuring latency using sparsednn, for {}'.format(target_layer_name))
        start = time.time()
        np.save('matrix.npy', weight)

        if N <= 4: continue

        ret_code = os.system("python fastsparse.py --row1 {} --col1 {} --col2 {}".format(M, K, N))
        if ret_code != 0:
            raise Exception(
                'latency measured failed for layer {}, ret_code:{}, M:{}, K:{}, N:{}'.format(target_layer_name,
                                                                                             ret_code, M, K, N))
        print('latency measurements obtained')
        expected_latency = np.load('matrix_latency.npy')
        end = time.time()
        print('used: {}s'.format(end - start))
        time_total += (end - start)

        total_latency += expected_latency.mean()

        layerwise_avg_latency.append(expected_latency.mean())
    print('total measuring cost: {}'.format(time_total))
    print(total_latency)
    os.chdir('..')
    return 1000 * total_latency, 1000 * np.array(layerwise_avg_latency)


def eval_sparsednn_c_api(model_state, inputs_dict, target_device, num_trials):
    import subprocess

    total_latency = 0
    layerwise_avg_latency = []
    print('measuring latency for model')
    time_total = 0
    start = time.time()

    MODULE_DIR = os.path.dirname(__file__)
    code_gen_path = os.path.join(MODULE_DIR, 'sparsednn', 'code_gen_cpu.py')
    asm_path = os.path.join(MODULE_DIR, 'sparsednn', 'test.s')
    cpp_path = os.path.join(MODULE_DIR, 'sparsednn', 'test.cpp')
    dylib_path = os.path.join(MODULE_DIR, 'sparsednn', 'test.so')
    executable_path = os.path.join(MODULE_DIR, 'sparsednn', 'test')
    matrix_transposed_path = os.path.join(MODULE_DIR, 'sparsednn', 'matrix_transposed.npy')
    bias_path = os.path.join(MODULE_DIR, 'sparsednn', 'bias.npy')
    bc_path = os.path.join(MODULE_DIR, 'sparsednn', 'BC.npy')
    cdriver_path = os.path.join(MODULE_DIR, 'sparsednn', 'driver_cpu.cpp')
    for idx, (target_layer_name, input_detail) in enumerate(inputs_dict.items()):
        weight_tensor = model_state[target_layer_name]
        if weight_tensor.dim() == 4:
            cout, cin, hk, wk = weight_tensor.shape
            M = cout
            K = cin * hk * wk
            N = input_detail.output_dim[-1] * input_detail.output_dim[-2]
            weight = weight_tensor.reshape(M, K).to(target_device).numpy()
        else:
            M = weight_tensor.shape[0]
            K = weight_tensor.shape[1]
            N = 1
            weight = weight_tensor.to(target_device).numpy()

        print('start measuring latency using sparsednn, for {}'.format(target_layer_name))

        np.save(matrix_transposed_path, weight.transpose().astype(np.float32))
        bias = np.random.normal(size=(M))
        input = np.random.normal(size=(K, N))
        np.save(bias_path, bias.astype(np.float32))
        np.save(bc_path, input.astype(np.float32))
        if N <= 4: continue

        latency = []
        # code generation
        ret_code = os.system(
            f'python {code_gen_path} --A_dim {M} --B_dim {K} --C_dim {N} --AT 6 --CT 2 --B_blocks 1 --C_blocks 1 --Gy 1 --infile {matrix_transposed_path} --outfile {cpp_path} --outfile_asm {asm_path} --x86 --no_relu --infile_bias {bias_path} --fuse'
        )
        if ret_code != 0:
            raise Exception(
                'code generation failed for layer {}, ret_code:{}, M:{}, K:{}, N:{}'.format(target_layer_name, ret_code,
                                                                                            M, K, N))

        # gcc
        ret_code = os.system(f"gcc -shared -g {asm_path} -o {dylib_path}")
        if ret_code != 0:
            raise Exception(
                'gcc failed for layer {}, ret_code:{}, M:{}, K:{}, N:{}'.format(target_layer_name, ret_code, M, K, N))

        # icc
        ret_code = os.system(
            f"icc -fopenmp -I . -qmkl -O3 -march=native -D AT=6 -D CT=2 -D C_Blocks=1 -DA_dim={M} -DINFILE={matrix_transposed_path} -D B_dim={K} -D C_dim={N} -D C_blocks=1 -D X86=1 -D MULTI=0 {cdriver_path} -lcnpy -o {executable_path} -std=c++17")
        if ret_code != 0:
            raise Exception(
                'icc compile failed for layer {}, ret_code:{}, M:{}, K:{}, N:{}'.format(target_layer_name,
                                                                                        ret_code, M, K, N))
        for _ in range(num_trials):
            b = subprocess.run(f"{executable_path}", shell=True, stdout=subprocess.PIPE)
            if b.returncode != 0:
                print(b.stdout)
                raise Exception(
                    'latency measured failed for layer {}, ret_code:{}, M:{}, K:{}, N:{}'.format(target_layer_name,
                                                                                                 b.returncode, M, K, N))

            lines = b.stdout.decode().split('\n')
            for i, line in enumerate(lines):
                if 'spmm microkernel' in line:
                    lat = float(lines[i + 1].split()[2])
                    latency.append(lat)

        total_latency += np.array(latency).mean()
        layerwise_avg_latency.append(np.array(latency).mean())
    end = time.time()
    time_total = end - start
    print('total measuring cost: {}'.format(time_total))
    print(total_latency)

    return total_latency, np.array(layerwise_avg_latency)


# def eval_sparsednn_c_api(model_state, inputs_dict, target_device, num_trials):
#     import subprocess
#
#     total_latency = 0
#     layerwise_avg_latency = []
#     print('measuring latency for model')
#     time_total = 0
#     start = time.time()
#     os.chdir('sparsednn')
#     for idx, (target_layer_name, input_detail) in enumerate(inputs_dict.items()):
#         weight_tensor = model_state[target_layer_name]
#         if weight_tensor.dim() == 4:
#             cout, cin, hk, wk = weight_tensor.shape
#             M = cout
#             K = cin * hk * wk
#             N = input_detail.output_dim[-1] * input_detail.output_dim[-2]
#             weight = weight_tensor.reshape(M, K).to(target_device).numpy()
#         else:
#             M = weight_tensor.shape[0]
#             K = weight_tensor.shape[1]
#             N = 1
#             weight = weight_tensor.to(target_device).numpy()
#
#         print('start measuring latency using sparsednn, for {}'.format(target_layer_name))
#
#         np.save('matrix_transposed.npy', weight.transpose().astype(np.float32))
#         bias = np.random.normal(size=(M))
#         input = np.random.normal(size=(K, N))
#         np.save('bias.npy', bias.astype(np.float32))
#         np.save('BC.npy', input.astype(np.float32))
#         if N <= 4: continue
#
#         latency = []
#         for _ in range(num_trials):
#             b = subprocess.run(f'bash test_sparsednn_c_api.sh {M} {K} {N} 0', shell=True, stdout=subprocess.PIPE)
#             if b.returncode != 0:
#                 print(b.stdout)
#                 raise Exception(
#                     'latency measured failed for layer {}, ret_code:{}, M:{}, K:{}, N:{}'.format(target_layer_name,
#                                                                                                  b.returncode, M, K, N))
#
#             lines = b.stdout.decode().split('\n')
#             for i, line in enumerate(lines):
#                 if 'spmm microkernel' in line:
#                     lat = float(lines[i + 1].split()[2])
#                     latency.append(lat)
#
#         total_latency += np.array(latency).mean()
#         layerwise_avg_latency.append(np.array(latency).mean())
#     end = time.time()
#     time_total = end - start
#     print('total measuring cost: {}'.format(time_total))
#     print(total_latency)
#     os.chdir('..')
#     return 1000 * total_latency, 1000 * np.array(layerwise_avg_latency)


def eval_torch_sparse_cuda(model_state, inputs_dict, target_device, num_trials):
    total_latency = 0
    from torch_sparse import spmm
    print('measuring latency for model')
    time_total = 0

    for idx, (target_layer_name, input_detail) in enumerate(inputs_dict.items()):
        weight_tensor = model_state[target_layer_name]
        if weight_tensor.dim() == 4:
            cout, cin, hk, wk = weight_tensor.shape
            M = cout
            K = cin * hk * wk
            N = input_detail.output_dim[-1] * input_detail.output_dim[-2]
            weight = weight_tensor.reshape(M, K).to(target_device).numpy()
        else:
            M = weight_tensor.shape[0]
            K = weight_tensor.shape[1]
            N = 1
            weight = weight_tensor.to(target_device).numpy()

        print('start measuring latency using torch_sparse, for {}'.format(target_layer_name))

        dense = torch.tensor(np.random.normal(size=(K, N))).to(target_device)
        index = torch.tensor(np.nonzero(weight)).to(target_device)
        value = torch.tensor(weight[np.nonzero(weight)]).to(target_device)

        latency = []
        for _ in range(num_trials):
            start = time.time()
            spmm(index, value, M, K, dense)
            end = time.time()
            latency.append(end - start)

        # print('used: {}s'.format(end - start))
        time_total += sum(latency)
        total_latency += np.array(latency).mean()
    print('total measuring cost: {}'.format(time_total))
    print(total_latency)
    return 1000 * total_latency


def eval_torch_sparse_cpu(model_state, inputs_dict, target_device, num_trials):
    total_latency = 0
    from torch_sparse import spmm
    print('measuring latency for model')
    time_total = 0

    for idx, (target_layer_name, input_detail) in enumerate(inputs_dict.items()):
        weight_tensor = model_state[target_layer_name]
        if weight_tensor.dim() == 4:
            cout, cin, hk, wk = weight_tensor.shape
            M = cout
            K = cin * hk * wk
            N = input_detail.output_dim[-1] * input_detail.output_dim[-2]
            weight = weight_tensor.reshape(M, K).to(target_device).numpy()
        else:
            M = weight_tensor.shape[0]
            K = weight_tensor.shape[1]
            N = 1
            weight = weight_tensor.to(target_device).numpy()

        print('start measuring latency using torch_sparse, for {}'.format(target_layer_name))

        dense = torch.tensor(np.random.normal(size=(K, N))).to(target_device)
        index = torch.tensor(np.nonzero(weight)).to(target_device)
        value = torch.tensor(weight[np.nonzero(weight)]).to(target_device)

        latency = []
        for _ in range(num_trials):
            start = time.time()
            spmm(index, value, M, K, dense)
            end = time.time()
            latency.append(end - start)

        # print('used: {}s'.format(end - start))
        time_total += sum(latency)
        total_latency += np.array(latency).mean()
    print('total measuring cost: {}'.format(time_total))
    print(total_latency)
    return 1000 * total_latency


def eval_mkl_sparse_cpu(model_state, inputs_dict, target_device, num_trials):
    total_latency = 0
    print('measuring latency for model')
    time_total = 0

    layerwise_avg_latency = []
    for idx, (target_layer_name, input_detail) in enumerate(inputs_dict.items()):
        weight_tensor = model_state[target_layer_name]
        if weight_tensor.dim() == 4:
            cout, cin, hk, wk = weight_tensor.shape
            M = cout
            K = cin * hk * wk
            N = input_detail.output_dim[-1] * input_detail.output_dim[-2]
            weight = weight_tensor.reshape(M, K).to(target_device).numpy()
        else:
            M = weight_tensor.shape[0]
            K = weight_tensor.shape[1]
            N = 1
            weight = weight_tensor.to(target_device).numpy()

        print('start measuring latency using mkl, for {}'.format(target_layer_name))

        dense = np.random.normal(size=(K, N)).astype(np.float32)
        sparse = scipy.sparse.csr_matrix(weight)

        latency = []
        for _ in range(num_trials + 2):
            start = time.time()
            sparse_dot_mkl.dot_product_mkl(sparse, dense)
            end = time.time()
            latency.append(end - start)
        latency = latency[0:]
        # print('latency',latency)
        # print('used: {}s'.format(end - start))
        time_total += sum(latency)
        total_latency += np.array(latency).mean()

        layerwise_avg_latency.append(np.array(latency).mean())

    print('total measuring cost: {}'.format(time_total))
    print(total_latency)

    return 1000 * total_latency, 1000 * np.array(layerwise_avg_latency)


def eval_block_sparse_cuda(model_state, inputs_dict, target_device, num_trials):
    return -1


PLATFORM_LATENCY_EVAL = {
    ('sparsednn', 'cpu'): eval_sparsednn,
    ('sparsednn_c_api', 'cpu'): eval_sparsednn_c_api,
    ('torch_sparse', 'cuda'): eval_torch_sparse_cuda,
    ('torch_sparse', 'cpu'): eval_torch_sparse_cpu,

    ('block_sparse', 'cuda'): eval_block_sparse_cuda,

    ('mkl', 'cpu'): eval_mkl_sparse_cpu
}
