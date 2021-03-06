import json
import os
import pickle

from bson import json_util
# import modules
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK

import cabascModel
import lcrModel
import lcrModelAlt
import lcrModelInverse
import svmModel
from HAABSA import lcrModelAlt_hierarchical_v4, lcrModelAlt_v4_fine_tune
# import parameter configuration and data paths
from config import *
from loadData import *


# If the script gives an error try adjusting loadData.loadHyperData percentage, it seems like the LCR-Rot-hop++.method
# cannot handle residual batches of one.

def main():
    fine_tune = True
    runs = 10
    n_iter = 15

    # Name, year, train size.
    book_domain = ["book", 2019, 2700]
    hotel_domain = ["hotel", 2015, 200]
    apex_domain = ["Apex", 2004, 250]
    camera_domain = ["Camera", 2004, 310]
    creative_domain = ["Creative", 2004, 540]
    nokia_domain = ["Nokia", 2004, 220]
    # domains = [book_domain, hotel_domain, apex_domain, camera_domain, creative_domain, nokia_domain]
    domains = [hotel_domain]

    for domain in domains:
        run_hyper(domain=domain[0], year=domain[1], size=domain[2], fine_tune=fine_tune, runs=runs, n_iter=n_iter)


def run_hyper(domain, year, size, fine_tune, runs, n_iter):
    path = "hyper_results/fine_tuning/" + domain + "/" + str(FLAGS.n_iter) + "/"
    if fine_tune:
        train_path = "data/programGeneratedData/BERT/" + domain + "/768_" + domain + "_train_" + str(
            year) + "_BERT_" + str(
            size) + ".txt"
    else:
        train_path = "data/programGeneratedData/BERT/" + domain + "/768_" + domain + "_train_" + str(year) + "_BERT.txt"

    FLAGS.source_domain = domain
    FLAGS.target_domain = domain
    FLAGS.year = year
    FLAGS.n_iter = n_iter

    FLAGS.train_path = train_path
    FLAGS.test_path = "data/programGeneratedData/BERT/" + FLAGS.target_domain + "/" + str(
        FLAGS.embedding_dim) + "_" + FLAGS.target_domain + "_test_" + str(FLAGS.year) + "_BERT.txt"
    FLAGS.train_embedding = "data/programGeneratedData/" + FLAGS.embedding_type + "_" + FLAGS.source_domain + "_" + str(
        FLAGS.year) + "_" + str(FLAGS.embedding_dim) + ".txt"
    FLAGS.test_embedding = "data/programGeneratedData/" + FLAGS.embedding_type + "_" + FLAGS.target_domain + "_" + str(
        FLAGS.year) + "_" + str(FLAGS.embedding_dim) + ".txt"

    train_size, test_size, train_polarity_vector, test_polarity_vector = loadHyperData(FLAGS, True)

    # Was 248, 0.87
    remaining_size = test_size
    accuracyOnt = 1.00

    # Define variable spaces for hyperopt to run over
    global eval_num
    global best_loss
    global best_hyperparams
    eval_num = 0
    best_loss = None
    best_hyperparams = None

    finetunespace = [
        hp.choice('learning_rate', [0.00001, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.07, 0.1]),
        hp.quniform('keep_prob', 0.25, 0.75, 0.1),
        hp.choice('momentum', [0.85, 0.9, 0.95, 0.99]),
        hp.choice('l2', [0.00001, 0.0001, 0.001, 0.01, 0.1]),
    ]
    lcrspace = [
        hp.choice('learning_rate', [0.001, 0.005, 0.02, 0.05, 0.06, 0.07, 0.08, 0.09, 0.01, 0.1]),
        hp.quniform('keep_prob', 0.25, 0.75, 0.1),
        hp.choice('momentum', [0.85, 0.9, 0.95, 0.99]),
        hp.choice('l2', [0.00001, 0.0001, 0.001, 0.01, 0.1]),
    ]
    cabascspace = [
        hp.choice('learning_rate', [0.001, 0.005, 0.02, 0.05, 0.06, 0.07, 0.08, 0.09, 0.01, 0.1]),
        hp.quniform('keep_prob', 0.25, 0.75, 0.01),
    ]
    svmspace = [
        hp.choice('c', [0.001, 0.01, 0.1, 1, 10, 100, 1000]),
        hp.choice('gamma', [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100])
    ]

    # Define objectives for hyperopt
    def lcr_objective(hyperparams):
        global eval_num
        global best_loss
        global best_hyperparams

        eval_num += 1
        (learning_rate, keep_prob, momentum, l2) = hyperparams
        print(hyperparams)

        l, pred1, fw1, bw1, tl1, tr1, _, _ = lcrModel.main(FLAGS.hyper_train_path, FLAGS.hyper_eval_path, accuracyOnt,
                                                           test_size, remaining_size, learning_rate, keep_prob,
                                                           momentum,
                                                           l2)
        tf.reset_default_graph()

        # Save training results to disks with unique filenames

        print(eval_num, l, hyperparams)

        if best_loss is None or -l < best_loss:
            best_loss = -l
            best_hyperparams = hyperparams

        result = {
            'loss': -l,
            'status': STATUS_OK,
            'space': hyperparams,
        }

        save_json_result(str(l), result)

        return result

    def lcr_inv_objective(hyperparams):
        global eval_num
        global best_loss
        global best_hyperparams

        eval_num += 1
        (learning_rate, keep_prob, momentum, l2) = hyperparams
        print(hyperparams)

        l, pred1, fw1, bw1, tl1, tr1 = lcrModelInverse.main(FLAGS.hyper_train_path, FLAGS.hyper_eval_path, accuracyOnt,
                                                            test_size, remaining_size, learning_rate, keep_prob,
                                                            momentum,
                                                            l2)
        tf.reset_default_graph()

        # Save training results to disks with unique filenames

        print(eval_num, l, hyperparams)

        if best_loss is None or -l < best_loss:
            best_loss = -l
            best_hyperparams = hyperparams

        result = {
            'loss': -l,
            'status': STATUS_OK,
            'space': hyperparams,
        }

        save_json_result(str(l), result)

        return result

    def lcr_alt_objective(hyperparams):
        global eval_num
        global best_loss
        global best_hyperparams

        eval_num += 1
        (learning_rate, keep_prob, momentum, l2) = hyperparams
        print(hyperparams)

        l, pred1, fw1, bw1, tl1, tr1 = lcrModelAlt.main(FLAGS.hyper_train_path, FLAGS.hyper_eval_path, accuracyOnt,
                                                        test_size, remaining_size, learning_rate, keep_prob, momentum,
                                                        l2)
        tf.reset_default_graph()

        # Save training results to disks with unique filenames

        print(eval_num, l, hyperparams)

        if best_loss is None or -l < best_loss:
            best_loss = -l
            best_hyperparams = hyperparams

        result = {
            'loss': -l,
            'status': STATUS_OK,
            'space': hyperparams,
        }

        save_json_result(str(l), result)

        return result

    def lcr_altv4_objective(hyperparams):
        global eval_num
        global best_loss
        global best_hyperparams

        eval_num += 1
        (learning_rate, keep_prob, momentum, l2) = hyperparams
        print(hyperparams)

        l, pred1, fw1, bw1, tl1, tr1 = lcrModelAlt_hierarchical_v4.main(FLAGS.hyper_train_path, FLAGS.hyper_eval_path,
                                                                        accuracyOnt,
                                                                        test_size, remaining_size, learning_rate,
                                                                        keep_prob,
                                                                        momentum, l2)
        tf.reset_default_graph()

        # Save training results to disks with unique filenames

        print(eval_num, l, hyperparams)

        if best_loss is None or -l < best_loss:
            best_loss = -l
            best_hyperparams = hyperparams

        result = {
            'loss': -l,
            'status': STATUS_OK,
            'space': hyperparams,
        }

        save_json_result(str(l), result)

        return result

    def lcr_fine_tune_objective(hyperparams):
        global eval_num
        global best_loss
        global best_hyperparams

        eval_num += 1
        (learning_rate, keep_prob, momentum, l2) = hyperparams
        print(hyperparams)

        l, pred1, fw1, bw1, tl1, tr1 = lcrModelAlt_v4_fine_tune.main(FLAGS.hyper_train_path, FLAGS.hyper_eval_path,
                                                                     accuracyOnt,
                                                                     test_size, remaining_size, learning_rate,
                                                                     keep_prob,
                                                                     momentum, l2)
        tf.reset_default_graph()

        # Save training results to disks with unique filenames

        print(eval_num, l, hyperparams)

        if best_loss is None or -l < best_loss:
            best_loss = -l
            best_hyperparams = hyperparams

        result = {
            'loss': -l,
            'status': STATUS_OK,
            'space': hyperparams,
        }

        save_json_result(str(l), result)

        return result

    def cabasc_objective(hyperparams):
        global eval_num
        global best_loss
        global best_hyperparams

        eval_num += 1
        (learning_rate, keep_prob) = hyperparams
        print(hyperparams)

        l = cabascModel.main(FLAGS.hyper_train_path, FLAGS.hyper_eval_path, accuracyOnt, test_size, remaining_size,
                             learning_rate, keep_prob)
        tf.reset_default_graph()

        # Save training results to disks with unique filenames

        print(eval_num, l, hyperparams)

        if best_loss is None or -l < best_loss:
            best_loss = -l
            best_hyperparams = hyperparams

        result = {
            'loss': -l,
            'status': STATUS_OK,
            'space': hyperparams,
        }

        save_json_result(str(l), result)

        return result

    def svm_objective(hyperparams):
        global eval_num
        global best_loss
        global best_hyperparams

        eval_num += 1
        (c, gamma) = hyperparams
        print(hyperparams)

        l = svmModel.main(FLAGS.hyper_svm_train_path, FLAGS.hyper_svm_eval_path, accuracyOnt, test_size, remaining_size,
                          c,
                          gamma)
        tf.reset_default_graph()

        # Save training results to disks with unique filenames

        print(eval_num, l, hyperparams)

        if best_loss is None or -l < best_loss:
            best_loss = -l
            best_hyperparams = hyperparams

        result = {
            'loss': -l,
            'status': STATUS_OK,
            'space': hyperparams,
        }

        save_json_result(str(l), result)

        return result

    # Run a hyperopt trial
    def run_a_trial():
        max_evals = nb_evals = 1

        print("Attempt to resume a past training if it exists:")

        try:
            # https://github.com/hyperopt/hyperopt/issues/267
            trials = pickle.load(open(path + "results.pkl", "rb"))
            print("Found saved Trials! Loading...")
            max_evals = len(trials.trials) + nb_evals
            print("Rerunning from {} trials to add another one.".format(
                len(trials.trials)))
        except:
            trials = Trials()
            print("Starting from scratch: new trials.")

        if fine_tune:
            objective = lcr_fine_tune_objective
            space = finetunespace
        else:
            objective = lcr_altv4_objective
            space = lcrspace

        best = fmin(
            # Insert the method objective function
            objective,  # lcr_altv4_objective/lcr_fine_tune_objective.
            # Define the methods hyper parameter space
            space=space,  # lcrspace/finetunespace.
            algo=tpe.suggest,
            trials=trials,
            max_evals=max_evals
        )
        pickle.dump(trials, open(path + "results.pkl", "wb"))

        print("\nOPTIMIZATION STEP COMPLETE.\n")
        print(best_hyperparams)

    def print_json(result):
        """Pretty-print a jsonable structure (e.g.: result)."""
        print(json.dumps(
            result,
            default=json_util.default, sort_keys=True,
            indent=4, separators=(',', ': ')
        ))

    def save_json_result(model_name, result):
        """Save json to a directory and a filename."""
        result_name = '{}.txt.json'.format(model_name)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, result_name), 'w') as f:
            json.dump(
                result, f,
                default=json_util.default, sort_keys=True,
                indent=4, separators=(',', ': ')
            )

    def load_json_result(best_result_name):
        """Load json from a path (directory + filename)."""
        result_path = os.path.join(path, best_result_name)
        with open(result_path, 'r') as f:
            return json.JSONDecoder().decode(
                f.read()
            )

    def load_best_hyperspace():
        results = [
            f for f in list(sorted(os.listdir(path))) if 'json' in f
        ]
        if len(results) == 0:
            return None

        best_result_name = results[-1]
        return load_json_result(best_result_name)["space"]

    def plot_best_model():
        """Plot the best model found yet."""
        space_best_model = load_best_hyperspace()
        if space_best_model is None:
            print("No best model to plot. Continuing...")
            return

        print("Best hyperspace yet:")
        print_json(space_best_model)

    for i in range(runs):
        print("Optimizing New Model")
        run_a_trial()
        # try:
        #    run_a_trial()
        # except Exception as err:
        #    err_str = str(err)
        #    print(err_str)
        #    traceback_str = str(traceback.format_exc())
        #    print(traceback_str)
        plot_best_model()


if __name__ == "__main__":
    main()
