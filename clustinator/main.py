import time
from datetime import datetime

from input import Input
from markovchain import MarkovChain
from clustering import Clustering
from analysis.cluster_analysis import Cluster_analysis as ca


"""states = ["INITIAL","login","View_Items","home","logout","View_Items_quantity","Add_to_Cart","shoppingcart",
          "remove","deferorder","purchasecart","inventory","sellinventory","clearcart","cancelorder","$"]"""


class Main:
    def __init__(self, sessions_file):
        self.sessions_file = sessions_file

    def start(self):
        """
        TODO: Summary for time, reuse past clustering
        """
        start_time = datetime.now()
        # Input data
        data_input = Input(self.sessions_file)
        epsilon, min_samples = data_input.cluster_param()

        print('load data done', datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
        start, stop = None, None
        step_size = 500
        iteration = 2

        for _ in enumerate(range(iteration)):
            print("Iteration:", _)
            if _ == (0, 0):
                pass
            else:
                start += step_size
                stop += step_size

            # Slice sessions @sessions(None, None)
            next_step, states = data_input.sessions(start, stop)

            if _ >= (1, 1):
                past_start = (start-step_size)
                past_stop = (stop-step_size)
                past_step, states = data_input.sessions(past_start, past_stop)

                # Compute transition matrix next
                mc_next = MarkovChain(next_step, states)
                mc_next = mc_next.csr_sparse_matrix()

                # Compute transition matrix past
                mc_past = MarkovChain(past_step, states)
                mc_past = mc_past.csr_sparse_matrix()

                print('matrix done', datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
                print('start clustering')

                # DBSCAN
                dbscan_next = Clustering(mc_next, epsilon, min_samples)
                dbscan_past = Clustering(mc_past, epsilon, min_samples)

                print("End clustering", datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), '\n')

                # BACKPROB HERE
                unique_next, counts_next, labels_next = dbscan_next.unique_labels()
                unique_past, counts_past, labels_past = dbscan_past.unique_labels()

                cluster_dict_next = dbscan_next.cluster_dict(labels_next, mc_next)
                cluster_dict_past = dbscan_past.cluster_dict(labels_past, mc_past)

                first_list = dbscan_next.list_cluster(cluster_dict_next, labels_next, labels_past)
                second_list = dbscan_past.list_cluster(cluster_dict_past, labels_next, labels_past)

                cluster_mean_hist = ca(first_list, second_list).cluster_backprob()

                #print(cluster_mean_hist)

        end_time = datetime.now()
        print('Duration: {}'.format(end_time - start_time))


if __name__ == '__main__':
    # Data imports
    PATH = "../poc/data/new_data/"
    sessions_file = (PATH + 'data.json')
    Main(sessions_file).start()
