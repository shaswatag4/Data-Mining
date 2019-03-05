import nflgame
import pandas as pd
import glob
import os
import matplotlib.pylab as plt


pd.set_option('display.expand_frame_repr', False)
pd.options.display.max_rows = 999

year = 2018

directory = './predictions/{}/'.format(year)
all_files = glob.glob(directory + '*.csv')

predictions_df = pd.concat((pd.read_csv(f) for f in all_files),
                           axis=0, sort=False, ignore_index=True)

success_counter = len(predictions_df.loc[predictions_df['success'] == 'Correct'])
game_count = len(predictions_df['game'])
success_pct = round((float(success_counter)/game_count)*100, 2)
print predictions_df
print '{}/{} {}%'.format(success_counter, game_count, success_pct)

plt.scatter(predictions_df['success'], predictions_df['prediction_probability'])
plt.show()
