from livedesk_mailing_list.data.faker import generate_data
from livedesk_mailing_list.utils.experiment import run_experiment

if __name__ == "__main__":

    # Generate a larger articles dataset
    n_articles = 1000
    n_users = 50
    articles, user_interactions = generate_data(n_articles, n_users)

    # Run Experiment
    run_experiment(articles, user_interactions, config_path='config.yaml')
