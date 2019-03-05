
class StatsGenerator:

    def __init__(self, year, weeks=None):

        import nflgame

        self.current_year, self.current_week = nflgame.live.current_year_and_week()
        self.year = year
        self.current_year_marker = (self.year == self.current_year)

        if weeks is None:
            if self.year == self.current_year:
                weeks = range(1, self.current_week+1)
            else:
                weeks = range(1, 18)

        self.weeks = weeks

        self.functions = {
            'result':
                {'func': self.game_result_stats, 'file_name': '0_game_result_stats'},
            'schedule':
                {'func': self.schedule_stats, 'file_name': '1_schedule_stats'},
            'record':
                {'func': self.record_stats, 'file_name': '2_record_stats'},
            'matchup':
                {'func': self.matchup_stats, 'file_name': '3_matchup_stats'},
            'point_differential':
                {'func': self.point_differential_stats, 'file_name': '4_point_differential_stats'},
            'turnover':
                {'func': self.turnover_stats, 'file_name': '5_turnover_stats'},
            'third_down_pct':
                {'func': self.third_down_pct_stats, 'file_name': '6_third_down_stats'}
        }

    def generate_stats(self):
        import os
        import glob
        import csv
        import utils
        import pandas as pd

        for week in self.weeks:

            print '\rGenerating stats for week {}'.format(week)

            directory = './data/game_data/{}/{}/'.format(self.year, week)
            if not os.path.exists(directory):
                os.makedirs(directory)

            games = utils.get_week_schedule(self.year, week)

            for f in self.functions.keys():
                func_dict = self.functions[f]
                func = func_dict['func']
                file_name = '{}.csv'.format(func_dict['file_name'])
                path = directory + file_name

                update_results_tag = (f == 'result' and
                                      self.current_year_marker and
                                      self.current_week == week)

                if not os.path.exists(path) or update_results_tag:
                    data = func(games)
                    with open(path, 'wb') as csv_file:
                        writer = csv.writer(csv_file)
                        for line in data:
                            writer.writerow(line)

            # Combining all files together
            all_files = glob.glob(directory + '*.csv')

            df = pd.concat((pd.read_csv(f) for f in all_files),
                           axis=1, sort=False)
            df.to_csv('./data/game_data/{}/week_{}_database.csv'.format(year, week), index=False)

        self.combine_yearly_data()

    def combine_yearly_data(self):

        import glob
        import pandas as pd

        directory = './data/game_data/{}/'.format(year)
        all_files = glob.glob('{}*.csv'.format(directory))

        if len(all_files) == 17:
            print 'Combining yearly data'

            df = pd.concat((pd.read_csv(f) for f in all_files),
                           axis=0, sort=False)
            df.to_csv('./data/game_data/{}_database.csv'.format(year), index=False)

        else:
            print 'Cannot combine, not enough week files for entire year'

    def schedule_stats(self, games):
        """
        Generates basic stats about the game which don't change throughout the season
        :return:
        """
        import utils

        header = ['gamekey', 'year', 'week',
                  'away', 'away_prev_wpct', 'away_prev_a_wpct',
                  'home', 'home_prev_wpct', 'home_prev_h_wpct',
                  'div_flag']

        data = []
        data.append(header)

        print 'Generating schedule stats'

        for game in games:
            # for convenience
            year, week = game['year'], game['week']
            home, away = game['home'], game['away']

            game['home_prev_wpct'] = utils.team_prev_season_win_pct(home, year, prev_seasons=1)
            game['home_prev_h_wpct'] = utils.team_prev_season_win_pct(home, year, prev_seasons=1,
                                                                      type='home')
            game['away_prev_wpct'] = utils.team_prev_season_win_pct(away, year, prev_seasons=1)
            game['away_prev_a_wpct'] = utils.team_prev_season_win_pct(away, year, prev_seasons=1,
                                                                      type='away')
            game['div_flag'] = utils.divisional_flag(home, away)

            row = [game[h] for h in header]
            data.append(row)

        return data

    def game_result_stats(self, games):
        """
        Generates stats for the result for each game if known, converting it into terms of a home win/loss,
        tie or unknown result (i.e game hasn't been played)
        :param games:
        :return:
        """
        import nflgame

        header = ['result']
        data = []
        data.append(header)

        print 'Generating game result stats'

        for game in games:
            # for convenience
            year, week = game['year'], game['week']
            home, away = game['home'], game['away']

            result_dictionary = {home: 'win', home + '/' + away: 'tie', away: 'loss'}

            try:
                winner = nflgame.one(year, week, home, away).winner
                result = result_dictionary[winner]
            except (AttributeError, KeyError):
                result = 'UNK'

            game['result'] = result

            row = [game[h] for h in header]
            data.append(row)

        return data

    def record_stats(self, games):
        """
        Generates stats regarding the home and away team's season records so far
        :param games:
        :return:
        """
        import utils

        header = ['away_record', 'away_wpct', 'away_a_wpct',
                  'home_record', 'home_wpct', 'home_h_wpct']
        data = []
        data.append(header)

        print 'Generating current record stats'

        for game in games:
            # for convenience
            year, week = game['year'], game['week']
            home, away = game['home'], game['away']

            # home record
            hw, hl, ht = utils.team_record(home, year, week - 1)
            game['home_record'] = '{}-{}-{}'.format(hw, hl, ht)

            # away record
            aw, al, at = utils.team_record(away, year, week - 1)
            game['away_record'] = '{}-{}-{}'.format(aw, al, at)

            game['home_wpct'] = utils.team_season_win_pct(home, year, week - 1)
            game['home_h_wpct'] = utils.team_season_win_pct(home, year, week - 1, type='home')

            game['away_wpct'] = utils.team_season_win_pct(away, year, week - 1)
            game['away_a_wpct'] = utils.team_season_win_pct(away, year, week - 1, type='away')

            row = [game[h] for h in header]
            data.append(row)

        return data

    def matchup_stats(self, games):
        """
        Generates stats of all the home vs away games in season so far and previous 2 seasons
        :param games:
        :return:
        """
        import utils

        header = ['matchup_weight']

        data_dict = {}
        data = []
        data.append(header)

        print 'Generating matchup stats'

        for game in games:
            data_dict['matchup_weight'] = utils.matchup_weight(game)
            row = [data_dict[h] for h in header]
            data.append(row)

        return data

    def point_differential_stats(self, games):
        """ Generates stats of each team's point differential over the previous season, seeason so far and last
        3 and 5 games
        :param games:
        :return:
        """
        import utils

        data_dict = {}
        data = []
        header = ['home_season_pt_dif', 'home_3game_pt_dif',
                  'home_5game_pt_dif', 'home_prev_season_pt_dif',
                  'away_season_pt_dif', 'away_3game_pt_dif',
                  'away_5game_pt_dif', 'away_prev_season_pt_dif']
        data.append(header)

        print 'Generating point differential stats'

        for game in games:
            # for convenience
            year, week = game['year'], game['week']
            home, away = game['home'], game['away']

            home_season_pt_dif, home_3game_pt_dif, home_5game_pt_dif = \
                utils.team_pt_dif_per_n_games(home, year, week)
            data_dict['home_season_pt_dif'] = home_season_pt_dif
            data_dict['home_3game_pt_dif'] = home_3game_pt_dif
            data_dict['home_5game_pt_dif'] = home_5game_pt_dif
            data_dict['home_prev_season_pt_dif'] = utils.team_pt_dif_per_game_season(home, year - 1)

            away_season_pt_dif, away_3game_pt_dif, away_5game_pt_dif = \
                utils.team_pt_dif_per_n_games(away, year, week)
            data_dict['away_season_pt_dif'] = away_season_pt_dif
            data_dict['away_3game_pt_dif'] = away_3game_pt_dif
            data_dict['away_5game_pt_dif'] = away_5game_pt_dif
            data_dict['away_prev_season_pt_dif'] = utils.team_pt_dif_per_game_season(away, year - 1)

            row = [data_dict[h] for h in header]
            data.append(row)

        return data

    def turnover_stats(self, games):
        """
        Generates turnover stats for each team so far that season, last season and last 2 and 5 games
        :param games:
        :return:
        """
        import utils

        data_dictionary = {}
        data = []
        header = [
            'home_season_turnovers',
            'home_3game_turnovers',
            'home_5game_turnovers',
            'home_prev_season_turnovers',
            'away_season_turnovers',
            'away_3game_turnovers',
            'away_5game_turnovers',
            'away_prev_season_turnovers',
            'home_season_turnover_dif',
            'home_3game_turnover_dif',
            'home_5game_turnover_dif',
            'home_prev_season_turnover_dif',
            'away_season_turnover_dif',
            'away_3game_turnover_dif',
            'away_5game_turnover_dif',
            'away_prev_season_turnover_dif',
        ]
        data.append(header)

        print 'Generating turnover stats'

        for game in games:
            # for convenience
            year, week = game['year'], game['week']
            home, away = game['home'], game['away']

            # Previous season stats
            home_prev_season_turnover_dict = utils.turnovers_per_game_season(home, year-1)
            away_prev_season_turnover_dict = utils.turnovers_per_game_season(away, year-1)
            data_dictionary['home_prev_season_turnovers'] = \
                home_prev_season_turnover_dict['turnovers_per_game']
            data_dictionary['home_prev_season_turnover_dif'] = \
                home_prev_season_turnover_dict['turnover_dif_per_game']
            data_dictionary['away_prev_season_turnovers'] = \
                away_prev_season_turnover_dict['turnovers_per_game']
            data_dictionary['away_prev_season_turnover_dif'] = \
                away_prev_season_turnover_dict['turnover_dif_per_game']

            for team, label in zip([home, away], ['home', 'away']):
                turnover_dict = utils.turnovers_per_game(team, year, week)
                data_dictionary[label + '_season_turnovers'] = \
                    turnover_dict['season_turnovers_per_game']
                data_dictionary[label + '_3game_turnovers'] = \
                    turnover_dict['3game_turnovers_per_game']
                data_dictionary[label + '_5game_turnovers'] = \
                    turnover_dict['5game_turnovers_per_game']
                data_dictionary[label + '_season_turnover_dif'] = \
                    turnover_dict['season_turnover_dif_per_game']
                data_dictionary[label + '_3game_turnover_dif'] = \
                    turnover_dict['3game_turnover_dif_per_game']
                data_dictionary[label + '_5game_turnover_dif'] = \
                    turnover_dict['5game_turnover_dif_per_game']

            row = [data_dictionary[h] for h in header]
            data.append(row)

        return data

    def third_down_pct_stats(self, games):
        """
        Generates turnover stats for each team
        :param games:
        :return:
        """
        import utils

        data_dictionary = {}
        headers = [
            'home_season_3down_pct',
            'home_3game_3down_pct',
            'home_5game_3down_pct',
            'away_season_3down_pct',
            'away_3game_3down_pct',
            'away_5game_3down_pct'
        ]
        data = []
        data.append(headers)

        print 'Generating third down stats'

        for game in games:
            # for convenience
            year, week = game['year'], game['week']
            home, away = game['home'], game['away']

            for team, label in zip([home, away], ['home', 'away']):
                tdp_dict = utils.third_down_pct_per_game(team, year, week)
                data_dictionary[label + '_season_3down_pct'] = \
                    tdp_dict['season_3down_pct_per_game']
                data_dictionary[label + '_3game_3down_pct'] = \
                    tdp_dict['3game_3down_pct_per_game']
                data_dictionary[label + '_5game_3down_pct'] = \
                    tdp_dict['5game_3down_pct_per_game']

            row = [data_dictionary[h] for h in headers]
            data.append(row)

        return data


if __name__ == '__main__':

    # weeks, year = [5, 6], 2017
    # stats = StatsGenerator(year)
    # stats.generate_stats()

    import nflgame

    years = range(2010, 2019)
    # years = [2018]

    for year in years:

        print year
        stats = StatsGenerator(year)
        stats.generate_stats()
