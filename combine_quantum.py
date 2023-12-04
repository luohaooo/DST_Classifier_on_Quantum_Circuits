# Combining BPAs on quantum circuits and conduct measurement
from qiskit import QuantumCircuit
from qiskit import Aer, transpile

def combine_qubits(circ, input1, input2, output):
    # x and y (same dimention) are input qubits to be combined by CCR
    # circ is the quantum circuits
    # z is the output qubits for the combined result
    qubits_num = len(input1)
    for qubit in range(qubits_num):
        circ.ccx(input1[qubit],input2[qubit],output[qubit])

def combineBPAs_quantum(alphas, shots):
    attributes_num = len(alphas)
    types_num = len(alphas[0])

    # establish quantum circuits
    qubits_num = (2*attributes_num-1)*types_num
    circ = QuantumCircuit(qubits_num)
    # build quantum BPAs
    for attribute in range(attributes_num):
        for type in range(types_num):
            circ.ry(2*alphas[attribute][type],attribute*types_num+type)
    # combine
    if attributes_num == 2:
        combine_qubits(circ, [ii for ii in range(types_num)], [ii+types_num for ii in range(types_num)], [ii+types_num*2 for ii in range(types_num)])
    else:
        combine_qubits(circ, [ii for ii in range(types_num)], [ii+types_num for ii in range(types_num)], [ii+types_num*attributes_num for ii in range(types_num)])
        for kk in range(2, attributes_num):
            combine_qubits(circ, [ii+types_num*kk for ii in range(types_num)], [ii+types_num*(kk+attributes_num-2) for ii in range(types_num)], [ii+types_num*(kk+attributes_num-1) for ii in range(types_num)])
    # measurement
    meas = QuantumCircuit(qubits_num, types_num)
    meas.barrier(range(qubits_num))
    # map the quantum measurement to the classical bits
    meas.measure(range(qubits_num-types_num,qubits_num), range(types_num))
    circ.add_register(meas.cregs[0])
    qc = circ.compose(meas)
    backend_sim = Aer.get_backend('qasm_simulator')
    job_sim = backend_sim.run(transpile(qc, backend_sim), shots=shots)
    result_sim = job_sim.result()
    counts = result_sim.get_counts(qc)
    # print(counts)

    # process the measurement result and output
    dimension = 2**types_num
    combined_BPA = [0 for item in range(dimension)]
    for key in counts.keys():
        value = counts[key]/shots
        index = int(key,2)
        combined_BPA[index] = value

    return combined_BPA