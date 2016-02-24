import os
import pandas as pd
import crimes_job_nn


def main(params):
    print params
    params = crimes_job_nn.get_params(params)
    print params

    loss_val, model, crimes, all_labels = crimes_job_nn.fit_model_and_test(params)

    pred_df = pd.DataFrame(model.predict_proba(crimes['features_test']), columns=all_labels)
    pred_df.to_csv(os.path.join(os.path.abspath(os.path.dirname(__file__)), './crimeSF_NN_logodds.csv'),
                   index_label="Id", na_rep="0")

    return loss_val


if __name__ == '__main__':

    # main({'layers': 1, 'hidden_units': 128, 'input_dropout': 0,
    #       'hidden_dropout': 0.344659, 'learning_rate': 0.05848, 'weight_decay': 0.000468})

    main({u'layers': 2, u'hidden_units': u'256', u'learning_rate': 0.025993748870243978,
          u'input_dropout': 0.42367816324422514, u'hidden_dropout': 0.0916930691866583,
          u'weight_decay': 0.0023796940697770894})

'''
loss_all:  2.19293237468
loss_train:  2.19113759238
loss_val:  2.19472238042
'''
