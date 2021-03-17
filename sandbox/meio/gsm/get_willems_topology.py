from sandbox.meio.gsm.analysis_utils import get_network

if __name__ == '__main__':
    for i in range(1, 39):
        print(i, get_network(i)[0])
