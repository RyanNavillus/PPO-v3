import pandas as pd


df = pd.read_csv('HNS.csv')


def calculate_hns(env_id, score):
    """
    Calculate the human normalized score for a given environment and score.

    Parameters
    ----------
    env_id : str
        The environment ID.
    score : float
        The score for the environment.

    Returns
    -------
    float
        The human normalized score.
    """
    # Get the mean and standard deviation for the environment
    human = float(df.loc[df['env_id'] == env_id, 'human'].values[0])
    record = float(df.loc[df['env_id'] == env_id, 'record'].values[0])
    random = float(df.loc[df['env_id'] == env_id, 'random'].values[0])

    # Calculate the human normalized score
    hns = (float(score) - random) / (human - random)
    rns = (float(score) - random) / (record - random)

    return hns, rns


if __name__ == "__main__":
    # Calculate the human normalized score for the Breakout-v5
    hns, rns = calculate_hns('Defender-v5', 200.0)
    print(hns, rns)
