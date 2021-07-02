import pandas as pd
import os 
import matplotlib.pyplot as plt

def get_values(all_results, tracker_name, column_names):

    
    print(100*round(all_results[tracker_name].loc[:,column_names],3))

def generate_box_plots(all_results, tracker_new_names=None):

    print(all_results)
    count_errors_relative = pd.DataFrame({tracker_name: pd.Series((results['IDs'][:-1]-results['GT_IDs'][:-1]))/results['GT_IDs'][:-1] \
        for tracker_name,results in all_results.items()})
    count_errors_relative.drop(labels=[29],inplace=True)
    # print(count_errors_relative)
    # print(count_errors_relative[24:].mean())
    # print(count_errors_relative[24:].std())

    count_errors_relative.columns = tracker_new_names
    count_errors_relative.boxplot()
    # plt.suptitle('Box plot on 17 independant short sequences from T1')
    plt.ylabel(r'$\frac{\hat{N}-N}{N}$')
    # plt.gca().get_xaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig('boxplot',format='pdf')
    # plt.show()

if __name__ == '__main__':

    eval_dir = 'data/trackers/surfrider_short_segments'
    tracker_names = ['fairmot_cleaned','sort','ours_cleaned']
    all_results = {tracker_name:pd.read_csv(os.path.join(eval_dir,'surfrider-test',tracker_name,'pedestrian_detailed.csv')) for tracker_name in tracker_names}

    # get_values(all_results,'fairmot',['AssRe___50','AssPr___50'])

    tracker_new_names = ['FairMOT (thresholded)','SORT','Ours (thresholded)']


    generate_box_plots(all_results=all_results, tracker_new_names=tracker_new_names)
