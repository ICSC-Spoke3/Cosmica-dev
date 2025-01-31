from pprint import pprint
from PyCosmica.parser import parse

if __name__ == '__main__':
    print('Start')
    sim_params = parse('/home/matteo.grazioso/Cosmica-dev/Cosmica_V7-speedtest/runned_tests/AMS-02_PRL2015/Input_Deuteron_TKO_20110509_20131121_r00100_lat00000.txt')
    pprint(dict(sim_params._asdict()), depth=10)
    # parse(
    # '/home/matteo.grazioso/Cosmica-dev/extra/simfiles/results/AMS-02Daily_20110802_Proton/inputs/Proton_1_117872e-04_20110730_20110730_r00100_lat00000.txt')