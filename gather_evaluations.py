import pandas as pd
import os 
import matplotlib.pyplot as plt
import pickle 
import numpy as np 
tracker_names = ['fairmot_cleaned','sort','ours_cleaned','ours']
eval_dir_part_1 = 'data/trackers/surfrider_part_1_only'
eval_dir_all = 'data/trackers/surfrider_long_segments'
eval_dir_short = 'data/trackers/surfrider_short_segments'
long_segments_names = ['part_1_1','part_1_2','part_2','part_3']

_round = lambda x: 100*round(x,3)


def get_det_values():

    results_p1_ours =  pd.read_csv(os.path.join(eval_dir_part_1,'surfrider-test','ours','pedestrian_detailed.csv'))
    all_results_ours = pd.read_csv(os.path.join(eval_dir_all,'surfrider-test','ours','pedestrian_detailed.csv'))
    
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

    print(f"{ass_re_p1}\n{ass_re_p2}\n{ass_re_p3}\n{ass_re_cb}")

    return [ass_re_p1,ass_re_p2,ass_re_p3,ass_re_cb]


def generate_box_plots(tracker_new_names=None):

    all_results = {tracker_name:pd.read_csv(os.path.join(eval_dir_short,'surfrider-test',tracker_name,'pedestrian_detailed.csv')) for tracker_name in ['fairmot_cleaned','sort','ours_cleaned']}

    print(all_results)
    count_errors = pd.DataFrame({tracker_name: pd.Series((results['IDs'][:-1]-results['GT_IDs'][:-1])) \
        for tracker_name,results in all_results.items()})
    # count_errors_relative.drop(labels=[29],inplace=True)

    print(count_errors)
    # fig, ax = plt.subplots(1,1,figsize=(10,10))
    count_errors.columns = tracker_new_names
    # ax = count_errors.boxplot(ax=ax)
    plt.plot(count_errors.T, linestyle='dashed')
    plt.boxplot(count_errors.T, positions=[0,1,2], labels=tracker_new_names)

    # plt.suptitle('Box plot on 17 independant short sequences from T1')
    plt.ylabel(r'$\hat{N}-N}$')
    # plt.gca().get_xaxis().set_visible(False)
    plt.tight_layout()
    # plt.savefig('boxplot.pdf',format='pdf')
    plt.show()


def get_count_err_mean_and_std_values(tracker_name):
    results_long_part_1 = pd.read_csv(os.path.join(eval_dir_part_1,'surfrider-test',tracker_name,'pedestrian_detailed.csv'))
    all_results_long = pd.read_csv(os.path.join(eval_dir_all,'surfrider-test',tracker_name,'pedestrian_detailed.csv'))
    all_results_shorts = pd.read_csv(os.path.join(eval_dir_short,'surfrider-test',tracker_name,'pedestrian_detailed.csv'))

    count_errors_part_1 = results_long_part_1['IDs'][2]-results_long_part_1['GT_IDs'][2]
    count_errors_part_2 = all_results_long['IDs'][2]-all_results_long['GT_IDs'][2]
    count_errors_part_3 = all_results_long['IDs'][3]-all_results_long['GT_IDs'][3]
    count_errors_combined = count_errors_part_1 + count_errors_part_2 + count_errors_part_3

    count_errors_shorts = pd.Series((all_results_shorts['IDs'][:-1]-all_results_shorts['GT_IDs'][:-1]))
    count_err_mean_p1 = round(count_errors_shorts[:16].mean(),2)
    count_err_std_p1 = round(count_errors_shorts[:16].std(),2)

    count_err_mean_p2 = round(count_errors_shorts[16:23].mean(),2)
    count_err_std_p2 = round(count_errors_shorts[16:23].std(),2)

    count_err_mean_p3 = round(count_errors_shorts[23:].mean(),2)
    count_err_std_p3 = round(count_errors_shorts[23:].std(),2)

    count_err_mean_cb = round(count_errors_shorts.mean(),2)
    count_err_std_cb = round(count_errors_shorts.std(),2)

    print(f"{count_errors_part_1} & {count_err_mean_p1} & {count_err_std_p1}\n{count_errors_part_2} & {count_err_mean_p2} & {count_err_std_p2}\n{count_errors_part_3} & {count_err_mean_p3} & {count_err_std_p3}\n{count_errors_combined} & {count_err_mean_cb} & {count_err_std_cb}\n")

def print_ass_re_for_trackers():

    ass_re_sort = get_ass_re_values('sort')
    ass_re_ours = get_ass_re_values('ours')
    ass_re_fairmot = get_ass_re_values('fairmot')

    plt.scatter(['Part 1', 'Part 2', 'Part 3', 'Combined'], ass_re_sort, marker='x', label='Sort')
    plt.plot(ass_re_sort[:-1], ls='--')
    plt.scatter(['Part 1', 'Part 2', 'Part 3', 'Combined'], ass_re_ours, marker='o', label='Ours')
    plt.plot(ass_re_ours[:-1], ls='--')
    plt.scatter(['Part 1', 'Part 2', 'Part 3', 'Combined'], ass_re_fairmot, marker='+',label='FairMOT*')
    plt.plot(ass_re_fairmot[:-1], ls='--')
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

def plot_framewise_TPs():
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
    
if __name__ == '__main__':

    # get_det_values()
    # get_ass_re_values('ours_cleaned')
    # get_count_err_mean_and_std_values('ours_cleaned')
    # generate_box_plots(tracker_new_names=['FairMOT*','SORT','Ours'])

    # print_ass_re_for_trackers()

    plot_framewise_TPs()