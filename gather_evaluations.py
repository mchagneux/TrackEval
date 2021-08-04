from itertools import count
import pandas as pd
import os 
import matplotlib.pyplot as plt
import pickle 
import numpy as np
from pandas.core.base import NoNewAttributesMixin 
import seaborn as sns 

_round = lambda x: 100*round(x,3)
eval_dir_part_1 = None
eval_dir_all = None
eval_dir_short = None
long_segments_names = None

def get_det_values(fps):

    results_p1_ours =  pd.read_csv(os.path.join(eval_dir_part_1,'surfrider-test',f'ours_{fps}_tau_0','pedestrian_detailed.csv'))
    all_results_ours = pd.read_csv(os.path.join(eval_dir_all,'surfrider-test',f'ours_{fps}_tau_0','pedestrian_detailed.csv'))
    
    det_re_p1 = _round(results_p1_ours.loc[2,'DetRe___50'])
    det_pr_p1 = _round(results_p1_ours.loc[2,'DetPr___50'])

    det_re_p2 = _round(all_results_ours.loc[2,'DetRe___50'])
    det_pr_p2 = _round(all_results_ours.loc[2,'DetPr___50'])

    det_re_p3 = _round(all_results_ours.loc[3,'DetRe___50'])
    det_pr_p3 = _round(all_results_ours.loc[3,'DetPr___50'])

    det_re_cb = _round(all_results_ours.loc[4,'DetRe___50'])
    det_pr_cb = _round(all_results_ours.loc[4,'DetPr___50'])

    print(f"{det_re_p1} & {det_pr_p1}\n{det_re_p2} & {det_pr_p2}\n{det_re_p3} & {det_pr_p3}\n{det_re_cb} & {det_pr_cb}\n")

def get_ass_re_values(tracker_name):

    results_p1 =   pd.read_csv(os.path.join(eval_dir_part_1,'surfrider-test',tracker_name,'pedestrian_detailed.csv'))
    all_results =  pd.read_csv(os.path.join(eval_dir_all,'surfrider-test',tracker_name,'pedestrian_detailed.csv'))

    ass_re_p1 = _round(results_p1.loc[2,'AssRe___50'])
    ass_re_p2 = _round(all_results.loc[2,'AssRe___50'])
    ass_re_p3 = _round(all_results.loc[3,'AssRe___50'])
    ass_re_cb = _round(all_results.loc[4,'AssRe___50'])

    return [ass_re_p1,ass_re_p2,ass_re_p3,ass_re_cb]

def generate_box_plots(tracker_new_names=None):

    all_results = {tracker_name:pd.read_csv(os.path.join(eval_dir_short,'surfrider-test',tracker_name,'pedestrian_detailed.csv')) for tracker_name in [f'ours_EKF_1_12fps_v0_tau_6', f'ours_EKF_1_smoothed_12fps_v0_tau_5']}

    # print(all_results)
    count_errors = pd.DataFrame({tracker_name: pd.Series((results['IDs'][:-1]-results['GT_IDs'][:-1])) \
        for tracker_name,results in all_results.items()})
    # count_errors_relative.drop(labels=[29],inplace=True)

    # print(count_errors)
    # fig, ax = plt.subplots(1,1,figsize=(10,10))
    count_errors.columns = tracker_new_names
    # ax = count_errors.boxplot(ax=ax)
    # plt.plot(count_errors.T, linestyle='dashed')
    # plt.boxplot(count_errors.T, positions=[0,1,2], labels=tracker_new_names)
    sns.pointplot(data=count_errors,ci="sd",estimator=np.mean, color='black',capsize=0.05)
    sns.swarmplot(data=count_errors)
    plt.hlines(y=0,linestyles='dashed',xmin=-1,xmax=4)
    # plt.suptitle('Box plot on 17 independant short sequences from T1')
    plt.ylabel(r'$err_s$')
    # plt.gca().get_xaxis().set_visible(False)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'boxplot_12fps_smoothed.pdf',format='pdf')
    # plt.savegi:()

def get_count_err_long(tracker_name):

    results_long_part_1 = pd.read_csv(os.path.join(eval_dir_part_1,'surfrider-test',tracker_name,'pedestrian_detailed.csv'))
    all_results_long = pd.read_csv(os.path.join(eval_dir_all,'surfrider-test',tracker_name,'pedestrian_detailed.csv'))

    count_errors_part_1 = results_long_part_1['IDs'][2]-results_long_part_1['GT_IDs'][2]
    count_errors_part_2 = all_results_long['IDs'][2]-all_results_long['GT_IDs'][2]
    count_errors_part_3 = all_results_long['IDs'][3]-all_results_long['GT_IDs'][3]
    count_errors_combined = count_errors_part_1 + count_errors_part_2 + count_errors_part_3


    return [count_errors_part_1, count_errors_part_2, count_errors_part_3, count_errors_combined]

def get_count_err_shorts(tracker_name):
    all_results_shorts = pd.read_csv(os.path.join(eval_dir_short,'surfrider-test',tracker_name,'pedestrian_detailed.csv'))

    return pd.Series((all_results_shorts['IDs'][:-1]-all_results_shorts['GT_IDs'][:-1]))

def get_count_err_mean_and_std_values(tracker_name):


    count_errors_part_1, count_errors_part_2, count_errors_part_3, count_errors_combined = get_count_err_long(tracker_name)
    count_errors_shorts = get_count_err_shorts(tracker_name)

    count_err_mean_p1 = round(count_errors_shorts[:16].mean(),2)
    count_err_std_p1 = round(count_errors_shorts[:16].std(),2)

    count_err_mean_p2 = round(count_errors_shorts[16:23].mean(),2)
    count_err_std_p2 = round(count_errors_shorts[16:23].std(),2)

    count_err_mean_p3 = round(count_errors_shorts[23:].mean(),2)
    count_err_std_p3 = round(count_errors_shorts[23:].std(),2)

    count_err_mean_cb = round(count_errors_shorts.mean(),2)
    count_err_std_cb = round(count_errors_shorts.std(),2)

    return [[count_err_mean_p1, count_err_std_p1],[count_err_mean_p2, count_err_std_p2],[count_err_mean_p3, count_err_std_p3],[count_err_mean_cb, count_err_std_cb]]

def print_ass_re_for_trackers(fps, tau):

    ass_re_sort = get_ass_re_values('sort')
    ass_re_ours = get_ass_re_values(f'ours_{fps}_{tau}')
    ass_re_fairmot = get_ass_re_values('fairmot_cleaned')

    plt.scatter(['Part 1', 'Part 2', 'Part 3', 'Combined'], ass_re_sort, marker='x', label='Sort')
    plt.plot(ass_re_sort, ls='--')
    plt.scatter(['Part 1', 'Part 2', 'Part 3', 'Combined'], ass_re_ours, marker='o', label='Ours')
    plt.plot(ass_re_ours, ls='--')
    plt.scatter(['Part 1', 'Part 2', 'Part 3', 'Combined'], ass_re_fairmot, marker='+',label='FairMOT*')
    plt.plot(ass_re_fairmot, ls='--')
    plt.legend()
    plt.ylabel('AssRe')
    plt.tight_layout()

    # plt.show()

    plt.savefig('ass_re_graphical.pdf',type='pdf')
   
def compute_new_fp_t(unmatched_tracker_ids):
    already_seen_tracker_ids = []
    new_fp_t = np.zeros(shape=(len(unmatched_tracker_ids)))
    for t,tracker_ids in enumerate(unmatched_tracker_ids):
        for tracker_id in tracker_ids: 
            if tracker_id not in already_seen_tracker_ids:
                new_fp_t[t]+=1
                already_seen_tracker_ids.append(tracker_id)
    return new_fp_t

def plot_framewise_TPs(fps):
    fps = 12
    framewise_results = dict()
    for tracker_name in tracker_names:
        framewise_results[tracker_name] = dict()
        for segment_name in long_segments_names:
            with open(os.path.join('framewise_results','{}_{}_framewise_results.pickle'.format(tracker_name,segment_name)),'rb') as f: 
                framewise_results[tracker_name][segment_name] = pickle.load(f)

    sequence_name = 'part_1_1'
    new_fp_t = dict()
    
    for tracker_name in tracker_names: 
        new_fp_t[tracker_name] = compute_new_fp_t(framewise_results[tracker_name][sequence_name][1][9])
    seconds = np.arange(len(new_fp_t['ours_cleaned']))/fps
    plt.plot(seconds, np.cumsum(new_fp_t['ours_cleaned']),label='Ours')
    plt.plot(seconds, np.cumsum(new_fp_t['fairmot_cleaned']),label='FairMOT*') 
    plt.plot(seconds, np.cumsum(new_fp_t['sort']),label='SORT')
    plt.legend()
    plt.show()          

def compare_tau_performance():
    
    tau_values = [0,1,2,3,4,5,6,7,8,9]

    counts = [get_count_err_long(f'ours_UKF_12fps_v0_tau_{tau}')[3] for tau in tau_values]
    # count_means, count_stds = np.array([get_count_err_mean_and_std_values(f'ours_UKF_12fps_v0_tau_{tau}')[-1] for tau in tau_values]).T

    fig, (ax0, ax1) = plt.subplots(1,2)

    ax0.scatter(tau_values, counts,c='black')
    ax0.set_xticks(tau_values,minor=True)
    ax0.hlines(y=0,linestyles='dashed',xmin=0,xmax=9)

    ax0.set_xlabel('$\\tau$')
    ax0.set_ylabel('$\hat{N}-N$')

    # ax1.errorbar(tau_values, count_means, count_stds, c='black')
    # ax1.hlines(y=0,linestyles='dashed',xmin=0,xmax=9)
    # ax1.set_xticks(tau_values,minor=True)

    # ax1.set_xlabel('$\\tau$')
    # ax1.set_ylabel('$\hat{\mu}$')

    ax1.set_axis_off()

    fig.tight_layout()
    plt.show()
    # plt.savefig('tau_study.pdf',format='pdf')
    
def compare_with_humans(human_counts_filename, tracker_names):


    tracker_results = []
    for tracker_name in tracker_names: 
        results_long_part_1 = pd.read_csv(os.path.join(eval_dir_part_1,'surfrider-test',tracker_name,'pedestrian_detailed.csv'))
        all_results_long = pd.read_csv(os.path.join(eval_dir_all,'surfrider-test',tracker_name,'pedestrian_detailed.csv'))
        gt_part_1 = results_long_part_1['GT_IDs'][2]
        count_part_1 = results_long_part_1['IDs'][2]

        gt_part_2 = all_results_long['GT_IDs'][2]
        count_part_2 = all_results_long['IDs'][2]

        gt_part_3 = all_results_long['GT_IDs'][3]
        count_part_3 = all_results_long['IDs'][3]
        tracker_results.append((count_part_1,count_part_2,count_part_3))
    
    tracker_results.append((gt_part_1,gt_part_2,gt_part_3))


    with open(human_counts_filename,'r') as f:
        human_counts = pd.read_csv(f,sep=',')
    
    human_counts = [human_counts.groupby('troncons').get_group(troncon).loc[:,'comptages'] for troncon in ['t1','t2','t3']]

    plt.boxplot(human_counts, labels=['Part 1','Part 2','Part 3'])
    plt.scatter([1,2,3],tracker_results[0], marker='x', label='FairMOT*')
    plt.scatter([1,2,3],tracker_results[1],  marker='o', label='SORT')
    plt.scatter([1,2,3],tracker_results[2],  marker='+', label='Ours')
    plt.scatter([1,2,3],tracker_results[3], marker='*',label='Count from video')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig('boxplot_humans.pdf',format='pdf')
    # plt.show()

def generate_table_values(tracker_name, new_name):

    ass_re_values = get_ass_re_values(tracker_name)
    count_mean_std = get_count_err_mean_and_std_values(tracker_name)
    count_errors = get_count_err_long(tracker_name)

    table = f"\\multirow{{ 3 }}{{*}}  {{{new_name}}}  &  Part 1  & {ass_re_values[0]} & {count_errors[0]} & {count_mean_std[0][0]} & {count_mean_std[0][1]} \\\ \n \\hhline{{~~~~~}}  &  Part 2  & {ass_re_values[1]} & {count_errors[1]} & {count_mean_std[1][0]} & {count_mean_std[1][1]} \\\ \n \hhline{{~~~~~}}  &  Part 3   & {ass_re_values[2]} & {count_errors[2]} & {count_mean_std[2][0]} & {count_mean_std[2][1]} \\\ \n \hhline{{~~~~~}}  &  Combined  & {ass_re_values[3]} & {count_errors[3]} & {count_mean_std[3][0]} & {count_mean_std[3][1]}\\\ \n   \hline \\[-1.8ex]"

    print(table)

if __name__ == '__main__':
    fps = 12
    tau = 'tau_6' if fps == 12 else 'tau_3'
    fps = f'{fps}fps'
    eval_dir_part_1 = 'external/TrackEval/data/trackers/surfrider_part_1_only_' + fps
    eval_dir_all = 'external/TrackEval/data/trackers/surfrider_long_segments_' + fps
    eval_dir_short = 'external/TrackEval/data/trackers/surfrider_short_segments_' + fps
    long_segments_names = ['part_1_1','part_1_2','part_2','part_3']
    # compare_with_humans('comptages_auterrive2021.csv',tracker_names=['fairmot_cleaned','sort',f'ours_{fps}_{tau}'])
    # get_det_values(fps)
    # get_ass_re_values(f'ours_{fps}_{tau}')
    # get_count_err_mean_and_std_values(f'ours_{fps}_{tau}')
    # generate_box_plots(tracker_new_names=['Ours','Ours, smoothed'])
#
    # # print_ass_re_for_trackers(fps, tau)

    # # plot_framewise_TPs()
    # generate_boxplots_to_compare_tau()

    # get_count_err_long('ours_EKF_1_12fps_v0_tau_3')
    # compare_tau_performance()
    # generate_table_values('ours_EKF_1_smoothed_12fps_v0_tau_5', new_name='$Filtering + Smoothing, \\tau=5$')
    # generate_table_values('ours_EKF_1_smoothed_12fps_v0_tau_6', new_name='$Filtering + Smoothing, \\tau=6$')
    # generate_table_values('ours_EKF_1_smoothed_12fps_v0_tau_7', new_name='$Filtering + Smoothing, \\tau=7$')

    # generate_table_values('ours_EKF_1_12fps_v0_tau_5', new_name='$Filtering, \\tau=5$')
    # generate_table_values('ours_EKF_1_12fps_v0_tau_6', new_name='$Filtering, \\tau=6$')
    # generate_table_values('ours_EKF_1_12fps_v0_tau_7', new_name='$Filtering, \\tau=7$')
    # print(get_count_err_long('ours_UKF_12fps_v0_tau_0'))
    print(get_count_err_long('ours_EKF_1_12fps_v0_tau_2'))
    print(get_count_err_long('ours_EKF_1_smoothed_12fps_v0_tau_2'))

    # get_ass_re_values('ours_EKF_order_1_12fps_tau_5')
