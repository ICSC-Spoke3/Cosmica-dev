from PyCosmica.parser import parse
from PyCosmica.propagation import propagation_vector

if __name__ == '__main__':
    print('Start')
    sim_params = parse(
        '/home/matteo.grazioso/Cosmica-dev/Cosmica_V7-speedtest/runned_tests/AMS-02_PRL2015/Input_Deuteron_TKO_20110509_20131121_r00100_lat00000.txt')
    # pprint(dict(sim_params._asdict()), depth=10)
    # sim_params = sim_params._replace(N_part=500)
    print(propagation_vector(sim_params.to_jit()))

    # parse(
    # '/home/matteo.grazioso/Cosmica-dev/extra/simfiles/results/AMS-02Daily_20110802_Proton/inputs/Proton_1_117872e-04_20110730_20110730_r00100_lat00000.txt')
