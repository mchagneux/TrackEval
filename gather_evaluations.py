import pandas as pd
import os 
import matplotlib.pyplot as plt
import pickle 
import numpy as np
import seaborn as sns 
from collections import defaultdict

_round_100 = lambda x: 100*round(x,2)
_round = lambda x: round(x,2)
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

def generate_box_plots(tracker_names, tracker_new_names=None):

    all_results = {tracker_name:pd.read_csv(os.path.join(eval_dir_short,'surfrider-test',tracker_name,'pedestrian_detailed.csv')) for tracker_name in tracker_names}

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

def plot_errors(tracker_names, tracker_new_names=None, last_index=-1):

    all_results = {tracker_name: pd.read_csv(os.path.join(eval_dir_short,'surfrider-test',tracker_name,'pedestrian_detailed.csv')) for tracker_name in tracker_names}

    predicted_counts = {k:v.loc[:,['Correct_IDs___50','Redundant_IDs___50','False_IDs___50','Missing_IDs___50']].iloc[:last_index] for k,v in all_results.items()}

    # predicted_counts['GT_IDs'] = all_results[tracker_names[0]].loc[:,['GT_IDs']].iloc[:-1]
    # print(all_results['sort'])
    # count_errors.index = all_results[tracker_names[0]]['seq'][:-1]
    # count_errors.columns = tracker_new_names
    # idxmins = count_errors.abs().idxmin(axis=1)
    


    # # count_errors_relative.drop(labels=[29],inplace=True)

    # # print(count_errors)
    # # fig, ax = plt.subplots(1,1,figsize=(10,10))

    positions = [len(predicted_counts.values())-x-3.5 for x in range(len(predicted_counts.values()))]

    fig, ax = plt.subplots()

    for (position, v) in zip(positions, predicted_counts.values()):
        v.index = all_results[tracker_names[0]]['seq'][:last_index]
        v.columns = ['Correct','Redundant','False','Missing']
        v.plot.bar(stacked=True, position=position, ax=ax, width=0.2, color=['green','orange','red','blue'], edgecolor='black',linewidth=0.1)
    
    gt_ids = all_results[tracker_names[0]].loc[:,['GT_IDs']].iloc[:last_index]
    # gt_ids.index = all_results[tracker_names[0]]['seq'][:last_index]
    gt_ids.columns = ['Ground truth']
    gt_ids.plot.bar(position=len(predicted_counts.values())-2.5,ax=ax,width=0.2,color='black')


    # # plt.vlines(x=[17,24],ymin=-10,ymax=10)
    # # plt.plot(idxmins)

    # plt.hlines(y=[0],xmin=-1,xmax=len(count_errors.index))
    # plt.ylabel('$err_s$')
    # plt.xlabel('$s$')
    # plt.xticks(np.arange(len(count_errors.index)),count_errors.index, rotation='vertical')
    # plt.grid(True,axis='y')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.xlabel('$s$')
    plt.ylabel('$Count$')

    plt.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.autoscale(True)
    # plt.show()
    plt.savefig('details_errors_ours_against_others.pdf',format='pdf')

def get_count_err_long(tracker_name):

    # results_long_part_1 = pd.read_csv(os.path.join(eval_dir_part_1,'surfrider-test',tracker_name,'pedestrian_detailed.csv'))
    all_results_long = pd.read_csv(os.path.join(eval_dir_all,'surfrider-test',tracker_name,'pedestrian_detailed.csv'))

    count_errors_part_1 = all_results_long[count_field][:2].sum() - all_results_long['GT_IDs'][:2].sum()
    count_errors_part_2 = all_results_long[count_field][2]-all_results_long['GT_IDs'][2]
    count_errors_part_3 = all_results_long[count_field][3]-all_results_long['GT_IDs'][3]
    count_errors_combined = count_errors_part_1 + count_errors_part_2 + count_errors_part_3

    return [count_errors_part_1, count_errors_part_2, count_errors_part_3, count_errors_combined]

def get_count_err_shorts(tracker_name):
    all_results_shorts = pd.read_csv(os.path.join(eval_dir_short,'surfrider-test',tracker_name,'pedestrian_detailed.csv'))
    return pd.Series((all_results_shorts[count_field][:-1]-all_results_shorts['GT_IDs'][:-1]))

def get_count_err_mean_and_std_values(tracker_name):


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

    counts = [get_count_err_long(f'ours_EKF_1_12fps_v0_tau_{tau}')[-1] for tau in tau_values]
    count_means, count_stds = np.array([get_count_err_mean_and_std_values(f'ours_EKF_1_12fps_v0_tau_{tau}')[-1] for tau in tau_values]).T

    fig, (ax0, ax1) = plt.subplots(1,2)

    ax0.scatter(tau_values, counts,c='black')
    ax0.set_xticks(tau_values,minor=True)
    ax0.hlines(y=0,linestyles='dashed',xmin=0,xmax=9)

    ax0.set_xlabel('$\\tau$')
    ax0.set_ylabel('$\hat{N}-N$')

    ax1.errorbar(tau_values, count_means, count_stds, c='black')
    ax1.hlines(y=0,linestyles='dashed',xmin=0,xmax=9)
    ax1.set_xticks(tau_values,minor=True)

    ax1.set_xlabel('$\\tau$')
    ax1.set_ylabel('$\hat{\mu}$')

    # ax1.set_axis_off()

    fig.tight_layout()
    if true_counts: plt.suptitle('Using true counts.')
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

def table_results_short(results, index_start=0, index_stop=-1):

    results = results.loc[:,['Correct_IDs___50','Redundant_IDs___50','False_IDs___50','Missing_IDs___50','Fused_IDs___50', 'GT_IDs']].iloc[index_start:index_stop]

    results.columns = ['correct','redundant','false','missing','mingled','gt']

    correct = results['correct']
    redundant = results['redundant']
    false = results['false']
    missing = results['missing']
    # mingled = results['mingled'] 
    gt = results['gt']

    diff_gt_correct = correct - gt 

    return [[f'{diff_gt_correct.mean():.1f}',f'{diff_gt_correct.std():.1f}'],
            [f'{false.mean():.1f}', f'{false.std():.1f}'],
            [f'{missing.mean():.1f}', f'{missing.std():.1f}'],
            [f'{redundant.mean():.1f}', f'{redundant.std():.1f}'],
            [f'{correct.sum()}', f'{false.sum()}',f'{missing.sum()}',f'{redundant.sum()}',f'{gt.sum()}']]

def get_table_values_averages(tracker_name, tracker_new_name):

    indices = [0,17,24,38]

    results_short_sequences = []

    results_short = pd.read_csv(os.path.join(eval_dir_short,'surfrider-test',tracker_name,'pedestrian_detailed.csv'))

    results_p1_long =   pd.read_csv(os.path.join(eval_dir_part_1,'surfrider-test',tracker_name,'pedestrian_detailed.csv'))
    results_long =  pd.read_csv(os.path.join(eval_dir_all,'surfrider-test',tracker_name,'pedestrian_detailed.csv'))

    ass_re_p1 = f"{100*results_p1_long.loc[2,'AssRe___50']:.1f}"
    ass_re_p2 = f"{100*results_long.loc[2,'AssRe___50']:.1f}"
    ass_re_p3 = f"{100*results_long.loc[3,'AssRe___50']:.1f}"
    # ass_re_cb = _round(results_long.loc[4,'AssRe___50'])

    
    for (index_start, index_stop) in zip(indices[:-1],indices[1:]):

        results_short_sequences.append(table_results_short(results_short, index_start, index_stop))
    
    results_short_sequences.append(table_results_short(results_short))

    table = f"\\multirow{{ 3 }}{{*}}  {{{tracker_new_name}}} & S1 & {ass_re_p1} & {results_short_sequences[0][0][0]} ({results_short_sequences[0][0][1]}) & {results_short_sequences[0][1][0]} ({results_short_sequences[0][1][1]}) & {results_short_sequences[0][2][0]} ({results_short_sequences[0][2][1]}) & {results_short_sequences[0][3][0]} ({results_short_sequences[0][3][1]}) \\\ \n"
    
    table += f"\\hhline{{~~~~~}}  &  S2  & {ass_re_p2} & {results_short_sequences[1][0][0]} ({results_short_sequences[1][0][1]}) & {results_short_sequences[1][1][0]} ({results_short_sequences[1][1][1]}) & {results_short_sequences[1][2][0]} ({results_short_sequences[1][2][1]}) & {results_short_sequences[1][3][0]} ({results_short_sequences[1][3][1]}) \\\ \n"

    table += f"\hhline{{~~~~~}}  &  S3  & {ass_re_p3} & {results_short_sequences[2][0][0]} ({results_short_sequences[2][0][1]}) & {results_short_sequences[2][1][0]} ({results_short_sequences[2][1][1]}) & {results_short_sequences[2][2][0]} ({results_short_sequences[2][2][1]}) & {results_short_sequences[2][3][0]} ({results_short_sequences[2][3][1]}) \\\ \n"
    
    table += f"\hhline{{~~~~~}} & Overall & - & {results_short_sequences[3][0][0]} ({results_short_sequences[3][0][1]}) & {results_short_sequences[3][1][0]} ({results_short_sequences[3][1][1]}) & {results_short_sequences[3][2][0]} ({results_short_sequences[3][2][1]}) & {results_short_sequences[3][3][0]} ({results_short_sequences[3][3][1]}) \\\ \n\hline \\\[-1.8ex]"

    print(table)

def get_table_values_absolute(tracker_names, tracker_new_names):
    
    indices = [0,17,24,38]

    table = f""
    for sequence_name, index_start, index_stop in zip(['S1','S2','S3'],indices[:-1],indices[1:]):
        results_for_sequence = []
        for i, tracker_name in enumerate(tracker_names):
            results_for_tracker = pd.read_csv(os.path.join(eval_dir_short,'surfrider-test',tracker_name,'pedestrian_detailed.csv'))
            results_for_sequence.append(table_results_short(results_for_tracker, index_start, index_stop)[-1])

        table += f"\\multirow{{ 3 }}{{*}}  {{{sequence_name}}}  &  {tracker_new_names[0]} & {results_for_sequence[0][0]} & {results_for_sequence[0][1]} & {results_for_sequence[0][2]} & {results_for_sequence[0][3]} \\\ \n" 
        table += f"\\hhline{{~~~~~}}   &  {tracker_new_names[1]} & {results_for_sequence[1][0]} & {results_for_sequence[1][1]} & {results_for_sequence[1][2]} & {results_for_sequence[1][3]} \\\ \n"
        table += f"\\hhline{{~~~~~}}   &  {tracker_new_names[2]} & {results_for_sequence[2][0]} & {results_for_sequence[2][1]} & {results_for_sequence[2][2]} & {results_for_sequence[2][3]} \\\ \n"
        table += f"\\hhline{{~~~~~}}  &  Ground truth & {results_for_sequence[0][4]} & 0 & 0 & 0 \\\ \n\hline \\\[-1.8ex] \n"

    print(table)

def read_mot_results_file(filename):
    raw_results =  np.loadtxt(filename, delimiter=',')
    if raw_results.ndim == 1: raw_results = np.expand_dims(raw_results,axis=0)
    tracklets = defaultdict(list) 
    for result in raw_results:
        track_id = int(result[1])
        frame_id = int(result[0])
        left, top, width, height = result[2:6]
        center_x = left + width/2
        center_y = top + height/2 
        tracklets[track_id].append((frame_id, center_x, center_y))

    tracklets = list(tracklets.values())

    return sorted(tracklets, key=lambda x:x[0][0])

def generate_nb_ids_in_track(track_starts, nb_ids_array):
    cnt=0
    for track_start in track_starts:
        cnt+=1
        nb_ids_array[track_start:] = cnt
    return nb_ids_array
    
def plot_evolutions_ids_for_file(tracker_names, tracker_new_names, sequences, sequence_name):

    if sequences == 'long':
        gt_dir = gt_dir_all
        eval_dir = eval_dir_all

    elif sequences == 'short':
        gt_dir = gt_dir_short
        eval_dir = eval_dir_short
    
    gt_filename = os.path.join(gt_dir,sequence_name,'gt','gt.txt')
    tracker_filenames = [os.path.join(eval_dir,'surfrider-test',tracker_name,'data',f'{sequence_name}.txt') for tracker_name in tracker_names]

    gt_tracks_starts = [track[0][0]-1 for track in read_mot_results_file(gt_filename)]
    trackers_tracks_starts = [[track[0][0]-1 for track in read_mot_results_file(tracker_filename)] for tracker_filename in tracker_filenames]

    last_frames_for_all = [gt_tracks_starts[-1]] + [tracker_tracks_starts[-1] for tracker_tracks_starts in trackers_tracks_starts]
    frame_nbs = np.arange(max(last_frames_for_all)+1)

    nb_ids_gt = generate_nb_ids_in_track(gt_tracks_starts, np.zeros_like(frame_nbs))
    nb_ids_trackers = [generate_nb_ids_in_track(tracker_tracks_starts, np.zeros_like(frame_nbs)) for tracker_tracks_starts in trackers_tracks_starts]


    plt.plot(nb_ids_gt, label='GT')
    for nb_ids_tracker, tracker_new_name in zip(nb_ids_trackers,tracker_new_names):
        plt.plot(nb_ids_tracker, label=tracker_new_name)
    plt.xlabel('Frame $n$')
    plt.ylabel('Number of objects up to frame $n$')
    plt.legend()
    plt.show()

if __name__ == '__main__':

    fps = 12
    tau = 'tau_6' if fps == 12 else 'tau_3'
    fps = f'{fps}fps'

    true_counts = True

    if true_counts: 
        count_field = 'True_IDs___10'
    else:
        count_field = 'IDs'

    gt_dir_short = f'external/TrackEval/data/gt/surfrider_short_segments_{fps}/surfrider-test' 
    gt_dir_all = f'external/TrackEval/data/gt/surfrider_long_segments_{fps}/surfrider-test' 

    eval_dir_part_1 = f'external/TrackEval/data/trackers/surfrider_part_1_only_{fps}' 
    eval_dir_all = f'external/TrackEval/data/trackers/surfrider_long_segments_{fps}' 
    eval_dir_short = f'external/TrackEval/data/trackers/surfrider_short_segments_{fps}' 

    long_segments_names = ['part_1_1','part_1_2','part_2','part_3']

    # get_table_values('fairmot','FairMOT')
    # get_table_values('fairmot_cleaned','FairMOT*')
    # get_table_values('sort','SORT')
    # get_table_values('ours_EKF_1_12fps_v0_tau_6','Ours')

    get_table_values_absolute(['fairmot_cleaned','sort','ours_EKF_1_12fps_v0_tau_6'],['FairMOT*','SORT','Ours'])

    # compare_with_humans('comptages_auterrive2021.csv',tracker_names=['fairmot_cleaned','sort',f'ours_{fps}_{tau}'])

    # plot_errors(['ours_EKF_1_12fps_v0_tau_6','sort','fairmot_cleaned'], last_index=17)
    # plot_errors(['ours_EKF_1_12fps_v0_tau_0','ours_EKF_1_12fps_v0_tau_6'])


    # generate_boxplots_to_compare_tau()

    # compare_tau_performance()

    # generate_table_values('ours_EKF_1_smoothed_12fps_v0_tau_5', new_name='$Filtering + Smoothing, \\tau=5$')
    # generate_table_values('fairmot_cleaned', new_name='$FairMOT*$')
    # generate_table_values('ours_EKF_1_smoothed_12fps_v0_tau_6', new_name='$Filtering + Smoothing, \\tau=6$')
    # generate_table_values('ours_EKF_1_smoothed_12fps_v0_tau_7', new_name='$Filtering + Smoothing, \\tau=7$')
    # generate_table_values('ours_EKF_1_12fps_v0_tau_6', new_name='$Filtering, \\tau=5$')
    # generate_table_values('fairmot','FairMOT')
    # generate_table_values('fairmot_cleaned','FairMOT*')
    # generate_table_values('sort', new_name='$SORT$')
    # generate_table_values('ours_EKF_1_12fps_v0_tau_0', new_name='$Ours_{\\tau=0}$')
    # generate_table_values('ours_EKF_1_12fps_v0_tau_6', new_name='$Ours_{\\tau=6}$')
    # generate_table_values('ours_EKF_1_12fps_v0_tau_7', new_name='$Filtering, \\tau=7$')



    # plot_evolutions_ids_for_file(tracker_names=['ours_EKF_1_12fps_v0_tau_7','sort'], tracker_new_names= ['Ours', 'SORT'], sequences='short',sequence_name='part_1_segment_0')