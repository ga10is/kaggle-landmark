import pandas as pd

from .. import config


def trans_label(s):
    s_new = s[3:]
    s_new = s_new.replace('/', '-')
    return s_new


def judge_landmark(i, splited, non_landmark_list):
    if splited[i * 2] in non_landmark_list:
        return ''
    else:
        return splited[i * 2] + ' ' + splited[i * 2 + 1]


def ajust_conf(s, non_landmark_list):
    splited = s.split(' ')
    if splited[0] in non_landmark_list:
        return ''
    half = int(len(splited) / 2)
    ajusted = [judge_landmark(i, splited, non_landmark_list)
               for i in range(half)]
    return ' '.join(ajusted)


def remove_non_landmark(submit):
    """
    Replace 'landmarks' column in submit dataframe to '' if the column is are predicted as non landmark.

    Parameters
    ----------
    submit: pd.DataFrame
        submit dataframe whichi has 'landmarks' column

    Returns
    -------
    submit: pd.DataFrame
        replaced dataframe
    """
    df = pd.read_csv(config.PLACES365_PATH)
    non_landmark = df[df['io'] == 1]
    non_landmark_list = non_landmark['label'].transform(trans_label)

    submit['landmarks'] = submit['landmarks'].transform(
        ajust_conf, non_landmark_list=non_landmark_list.tolist())
    # s_label = submit['landmarks'].str.split(' ', expand=True)[0]
    # is_non_landmark = s_label.isin(non_landmark_list)
    # submit.loc[is_non_landmark, 'landmarks'] = ''

    return submit
