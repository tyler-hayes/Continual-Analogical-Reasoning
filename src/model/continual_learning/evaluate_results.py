import json
import numpy as np
import os
import matplotlib.pyplot as plt


def compute_metrics(Rmat, offline=None):
    # given the R matrix of results and an offline learner's performance, compute all performance metrics
    def fwt_computation(Rmatrix):
        N = Rmatrix.shape[0]
        fwt = 0
        for j in range(N):
            fwt += sum([Rmatrix[i, j] for i in range(j)])
        fwt /= N * (N - 1) / 2
        return fwt

    def bwt_computation(Rmatrix):
        N = Rmatrix.shape[0]
        bwt = 0
        for i in range(1, N):
            bwt += sum([Rmatrix[i, j] - Rmatrix[j, j] for j in range(i - 1)])
        bwt /= N * (N - 1) / 2
        return bwt

    def avg_acc_computation(Rmatrix):
        N = Rmatrix.shape[0]
        acc = 0
        for j in range(N):
            acc += sum([Rmatrix[i, j] for i in range(j, N)])
        acc /= N * (N + 1) / 2
        return acc

    def gamma_t_computation(Rmatrix):
        mean_run = 0
        T = Rmatrix.shape[0]
        for i in range(T):
            acc = sum([Rmatrix[i, k] for k in range(T)]) / T
            mean_run += acc
        return mean_run / T

    def omega_computation(Rmatrix, offline):
        mean_run = 0
        T = Rmatrix.shape[0]
        for i in range(T):
            acc = sum([Rmatrix[i, k] for k in range(T)]) / T
            mean_run += acc / offline[i]
        return mean_run / T

    Rmat = Rmat.T
    fwt_score = fwt_computation(Rmat)
    bwt_score = bwt_computation(Rmat)
    avg_acc_score = avg_acc_computation(Rmat)

    if offline is not None:
        offline = offline / 100
        gamma_t = omega_computation(Rmat, offline)
    else:
        gamma_t = gamma_t_computation(Rmat)

    return avg_acc_score, fwt_score, bwt_score, gamma_t


def get_results(save_path, file_name='incremental_raven_accuracies.json'):
    # load saved results from disc
    def compute_average(d):
        avg = []
        for _, v in d.items():
            avg.append(v)
        return np.array(avg).mean(0), np.array(avg).mean(1)

    with open(os.path.join(save_path, file_name)) as file:
        results = json.load(file)
    with open(os.path.join(save_path, 'total_time.json')) as f:
        times = json.load(f)
    results['avg'], results['per_task_avg'] = compute_average(results)
    results['time'] = times
    return results


def load_partial_replay_results(replay_types, pretty_names, base_tasks, expt_name, suffix, samples, results_dir):
    # load saved partial replay results from disc
    results_dict = {}
    matrix_dict = {}
    for r, pretty in zip(replay_types, pretty_names):
        m_curve = []
        m_res = []
        times = []
        for b in base_tasks:
            rs = r + suffix
            expt = expt_name % (rs, samples, b)
            res = get_results(os.path.join(results_dir, expt))
            m_res.append(res)
            mu = np.array(res['avg'])
            m_curve.append(mu)
            times.append(res['time'])
        m_curve = np.array(m_curve)
        times = np.array(times)
        results_dict[pretty] = (m_curve, times)
        matrix_dict[pretty] = m_res
    return results_dict, matrix_dict


def load_baseline_expts(results_dir, expt_name, models, base_tasks):
    # load saved baseline results from disc
    results_dict = {}
    matrix_dict = {}
    for (m, reg, pretty) in models:
        m_curve = []
        m_res = []
        times = []
        for b in base_tasks:
            expt = expt_name % (m, reg, b)
            res = get_results(os.path.join(results_dir, expt))
            m_res.append(res)
            mu = np.array(res['avg'])
            m_curve.append(mu)
            times.append(res['time'])
        m_curve = np.array(m_curve)
        times = np.array(times)
        results_dict[pretty] = (m_curve, times)
        matrix_dict[pretty] = m_res
    return results_dict, matrix_dict


def check_regularization_model_grid_search(results_dir, expt_name, models, base_tasks, reg_params):
    # print performance of each regularization model for all hyper-parameters tested in grid search
    for m in models:
        for reg in reg_params:
            m_mu = []
            m_final = []
            for b in base_tasks:
                expt = expt_name % (m, reg, b)
                res = get_results(os.path.join(results_dir, expt))
                mu = np.array(res['avg'])
                m_mu.append(np.mean(mu))
                m_final.append(mu[-1])
            m_mu = float(np.mean(np.array(m_mu)))
            m_final = float(np.mean(np.array(m_final)))
            print('%s -- lambda=%d -- mean=%0.2f -- final=%0.2f' % (m, reg, m_mu, m_final))


def plot_baselines(results_dict, offline, save_dir=None, save_name='baselines_learning_curve', plot_std=False):
    # plot baseline learning curve
    markers = ['p', 's', 'D', '>', '^', '<', 'd', 'o', 'x']
    colors = ['#999999', '#377eb8', '#4daf4a',
              '#f781bf', '#a65628', '#984ea3',
              '#ff7f00', '#e41a1c', '#dede00']

    x = np.arange(1, 8, 1)
    mu_offline = np.mean(offline, axis=0)
    std_offline = np.std(offline, axis=0) / np.sqrt(offline.shape[0])

    fig, ax = plt.subplots()
    for i, (k, v) in enumerate(results_dict.items()):
        if k == 'offline':
            continue
        v = v[0]
        mu = np.mean(v, axis=0)
        num_runs = v.shape[0]
        std = np.std(v, axis=0) / np.sqrt(num_runs)
        ax.plot(x, mu, marker=markers[i], color=colors[i], linewidth=2, markersize=7, label=k)
        if plot_std:
            plt.fill_between(x, mu - std, mu + std, color=colors[i], alpha=0.5)

    # offline results
    ax.plot(x, mu_offline, 'kx--', linewidth=2, markersize=7, label='Offline')
    if plot_std:
        plt.fill_between(x, mu_offline - std_offline, mu_offline + std_offline, color='k', alpha=0.5)

    plt.xlabel('Number of Tasks Trained', fontweight='bold', fontsize=16)
    plt.ylabel('Accuracy [%]', fontweight='bold', fontsize=16)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim([0, 100])
    plt.legend(fontsize=12, loc='upper center', bbox_to_anchor=[0.5, 1.31], ncol=2, fancybox=True, shadow=True)
    plt.grid()
    plt.show()

    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, save_name + '.png'), bbox_inches="tight", format='png')


def get_omega_dict(results_dict, offline):
    # compute omega results for each method
    def compute_omega(model, offline):
        model = np.array(model)
        offline = np.array(offline)
        omega = np.mean(model / offline)
        return omega

    d = {}
    for i, (k, v) in enumerate(results_dict.items()):
        v = v[0]
        num_runs = v.shape[0]
        o = []
        for j in range(num_runs):
            o.append(compute_omega(v[j], offline[j]))
        omega = np.mean(np.array(o))
        d[k] = omega
    return d


def plot_sample_size_expt(replay_types, pretty_names, base_tasks, expt_name, results_dir, offline, save_dir,
                          save_name='partial_rehearsal_sample_size_expt'):
    # plot ablation study showing Omega performance as a function of number of replay samples
    omega_dict = {}
    for p in pretty_names:
        omega_dict[p] = []

    sample_sizes = [8, 16, 32, 64]
    for samples in sample_sizes:
        results_dict, _ = load_partial_replay_results(replay_types, pretty_names, base_tasks, expt_name, '',
                                                      samples, results_dir)
        d = get_omega_dict(results_dict, offline)
        for p in pretty_names:
            omega_dict[p].append(d[p])

    markers = ['p', 's', 'D', '>', '^', '<', 'd', 'o', 'x']
    colors = ['#999999', '#377eb8', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#ff7f00', '#e41a1c', '#dede00']

    fig, ax = plt.subplots()
    for i, (k, v) in enumerate(omega_dict.items()):
        ax.plot(sample_sizes, v, marker=markers[i], color=colors[i], linewidth=2, markersize=7, label=k)

    plt.xlabel('Number of Replay Samples', fontweight='bold', fontsize=16)
    plt.xscale('symlog', base=2)
    plt.ylabel(r'$\Omega$', fontweight='bold', fontsize=16)
    plt.xticks(sample_sizes, [r'$2^{3}$', r'$2^{4}$', r'$2^{5}$', r'$2^{6}$'], fontsize=18)
    plt.yticks([0.80, 0.85, 0.90, 0.95, 1.00], fontsize=18)
    plt.legend(fontsize=12, loc='upper center', bbox_to_anchor=[0.5, 1.31], ncol=2, fancybox=True, shadow=True)
    plt.grid()
    plt.show()

    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, save_name + '.png'), bbox_inches="tight", format='png')


def compute_Rmatrix_metrics(matrix_dict, results_dict, orders, offline=None):
    # make dictionary of R matrices to compute metrics
    R_dict = {}
    for k, v in matrix_dict.items():
        a = np.zeros((len(orders), 7, 7))  # 7 total tasks in raven
        for i, o in enumerate(orders):
            for j, t in enumerate(o):
                m = v[i][t]
                a[i][j, :] = m  # put performance for task as row in 'a' numpy array
        R_dict[k] = a / 100

    # compute Omega, Avg-Acc, Final-Acc, BWT, FWT, and Time results
    for k, v in R_dict.items():

        results_vals = results_dict[k]
        mu_final = float(np.mean([i[-1] for i in results_vals[0]]) / 100)
        mu_time = float(np.round(np.mean(results_vals[1]) / 60))

        avg_acc_list, fwt_list, bwt_list, gamma_t_list = [], [], [], []
        for i in range(v.shape[0]):
            if offline is not None:
                avg_acc_scores, fwt_scores, bwt_scores, gamma_t = compute_metrics(v[i, :, :], offline=offline[i, :])
            else:
                avg_acc_scores, fwt_scores, bwt_scores, gamma_t = compute_metrics(v[i, :, :], offline=offline)
            avg_acc_list.append(avg_acc_scores)
            fwt_list.append(fwt_scores)
            bwt_list.append(bwt_scores)
            gamma_t_list.append(gamma_t)

        # compute mean over runs
        mu_avg_acc = float(np.mean(np.array(avg_acc_list)))
        mu_fwt = float(np.mean(np.array(fwt_list)))
        mu_bwt = float(np.mean(np.array(bwt_list)))
        mu_gamma_t = float(np.mean(np.array(gamma_t_list)))

        print(
            '%s: Omega=%0.3f -- Avg-Acc=%0.3f -- Final-Acc=%0.3f -- BWT=%0.3f -- FWT=%0.3f -- Time(min)=%0.f' % (
                k, mu_gamma_t, mu_avg_acc, mu_final, mu_bwt, mu_fwt, mu_time))
    return R_dict


def get_partial_replay_results(replay_types, pretty_names, base_tasks, expt_name, results_dir, orders, offline,
                               save_dir=None, main_replay_samples=32):
    # make ablation plot of Omega performance as a function of number of replay samples
    plot_sample_size_expt(replay_types, pretty_names, base_tasks, expt_name, results_dir, offline, save_dir)

    # compute performance for balanced partial replay methods
    print('\n\nBalanced Partial Replay %d Sample Results:' % main_replay_samples)
    results_dict, matrix_dict = load_partial_replay_results(replay_types, pretty_names, base_tasks, expt_name,
                                                            suffix='_bal_oversample', samples=main_replay_samples,
                                                            results_dir=results_dir)
    compute_Rmatrix_metrics(matrix_dict, results_dict, orders, offline=offline)

    # compute performance for unbalance partial replay methods
    print('\n\nUnbalanced Partial Replay %d Sample Results:' % main_replay_samples)
    results_dict, matrix_dict = load_partial_replay_results(replay_types, pretty_names, base_tasks, expt_name,
                                                            suffix='', samples=main_replay_samples,
                                                            results_dir=results_dir)
    compute_Rmatrix_metrics(matrix_dict, results_dict, orders, offline=offline)

    # since max replays performs the best, we will store its performance for our main results
    # note: we assume max replays is final method in results_dict and matrix_dict
    max_replays_results_dict = results_dict[pretty_names[-1]]
    max_replays_matrix_dict = matrix_dict[pretty_names[-1]]
    return max_replays_results_dict, max_replays_matrix_dict


def get_offline_array():
    # store offline results here for convenience
    final = 91.7

    # these are from the individual offline runs
    cs_vals = np.array([26.52352608, 39.2786, 55.9311, 63.1378, 78.3588, 83.2129, final])
    ud_vals = np.array([29.11706349, 40.7171, 54.4289, 62.748, 70.5286, 81.3209, final])
    d4_vals = np.array([24.63151927, 42.8359, 57.6743, 61.6567, 69.0972, 81.9799, final])

    arr = np.concatenate(
        [np.expand_dims(cs_vals, axis=0), np.expand_dims(ud_vals, axis=0), np.expand_dims(d4_vals, axis=0)], axis=0)
    return arr


def main():
    ### CHANGE THESE PATH PARAMETERS ###
    base_dir = '/media/tyler/Data/codes/Continual-Analogical-Reasoning'
    results_dir = base_dir + '/src/model/continual_learning/analogical_reasoning_results'
    save_dir = base_dir + '/src/model/continual_learning/analogical_reasoning_plots'

    ###################################################################################################################
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    base_tasks = ['cs', 'ud', 'd4']
    orders = [['cs', 'io', 'lr', 'ud', 'd4', 'd9', '4c'],
              ['ud', 'cs', 'io', '4c', 'd9', 'd4', 'lr'],
              ['d4', 'lr', '4c', 'ud', 'd9', 'cs', 'io']]
    ###################################################################################################################
    # partial replay parameters
    replay_types = ['random', 'logit_dist_proba_shift_min', 'confidence_proba_shift_min', 'margin_proba_shift_min',
                    'time_proba_shift_min', 'loss_proba_shift_min', 'replay_count_proba_shift_min']
    replay_pretty_names = ['Random', 'Min Logit Dist', 'Min Confidence', 'Min Margin', 'Max Time', 'Max Loss',
                           'Min Replays']
    partial_replay_expt_name = 'incremental_partial_replay_strategy_%s_raven_samples_%d_base_task_%s'
    ###################################################################################################################
    # main baseline parameters
    baseline_expt_name = 'incremental_%s_lambda_%d_raven_base_task_%s'
    baseline_models = [('fine_tune', 1, 'Fine-Tune (Stream)'),
                       ('fine_tune_batch', 1, 'Fine-Tune (Batch)'),
                       ('distillation', 1, 'Distillation (Batch)'),
                       ('ewc', 10, 'EWC (Batch)'),
                       ('cumulative_replay', 1, 'Cumulative Replay (Batch)')]
    baseline_pretty_names = ['Fine-Tune (Stream)', 'Fine-Tune (Batch)', 'Distillation (Batch)', 'EWC (Batch)',
                             'Partial Replay (Stream)', 'Cumulative Replay (Batch)']
    ###################################################################################################################
    # grab offline performance for normalization
    offline = get_offline_array()
    ###################################################################################################################
    # gather all partial replay results and return "best" model for main baselines
    max_replays_results_dict, max_replays_matrix_dict = get_partial_replay_results(replay_types, replay_pretty_names,
                                                                                   base_tasks, partial_replay_expt_name,
                                                                                   results_dir, orders, offline,
                                                                                   save_dir)
    ###################################################################################################################
    print('\n\nBaseline Models Results:')

    results_dict, matrix_dict = load_baseline_expts(results_dir, baseline_expt_name, baseline_models, base_tasks)

    # add best partial replay model to main results
    results_dict['Partial Replay (Stream)'] = max_replays_results_dict
    matrix_dict['Partial Replay (Stream)'] = max_replays_matrix_dict

    # for appearance, let's make cumulative replay model last to be plot since it is the best performer
    l = sorted(results_dict.items(), key=lambda pair: baseline_pretty_names.index(pair[0]))
    results_dict_sorted = {}
    for (k, v) in l:
        results_dict_sorted[k] = v

    matrix_dict_sorted = {}
    for (k, v) in l:
        matrix_dict_sorted[k] = matrix_dict[k]

    # compute baseline metrics and make learning curve plot
    compute_Rmatrix_metrics(matrix_dict_sorted, results_dict_sorted, orders, offline=offline)
    plot_baselines(results_dict_sorted, offline, save_dir)
    # ###################################################################################################################
    # compute results for regularization model hyperparameter grid search
    print('\n\nLambda Tuning Regularization Results:')
    expt_name = 'incremental_%s_grid_search_lambda_%d_raven_base_task_%s'
    baseline_models = ['ewc', 'distillation']
    reg_params = [1, 10, 100]
    check_regularization_model_grid_search(results_dir, expt_name, baseline_models, base_tasks, reg_params)


if __name__ == '__main__':
    main()
