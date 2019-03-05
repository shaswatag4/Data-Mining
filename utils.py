import nflgame
import numpy as np
import pandas as pd

division_dictionary = {
    'ARI': 'NFC_west',
    'ATL': 'NFC_south',
    'BAL': 'AFC_north',
    'BUF': 'AFC_east',
    'CAR': 'NFC_south',
    'CHI': 'NFC_north',
    'CIN': 'AFC_north',
    'CLE': 'AFC_north',
    'DAL': 'NFC_east',
    'DEN': 'AFC_west',
    'DET': 'NFC_north',
    'GB':  'NFC_north',
    'HOU': 'AFC_south',
    'IND': 'AFC_south',
    'JAC': 'AFC_south',
    'JAX': 'AFC_south',
    'KC':  'AFC_west',
    'STL': 'NFC_west',
    'LA':  'NFC_west',
    'SD':  'AFC_west',
    'LAC': 'AFC_west',
    'MIA': 'AFC_east',
    'MIN': 'NFC_north',
    'NE':  'AFC_east',
    'NO':  'NFC_south',
    'NYG': 'NFC_east',
    'NYJ': 'AFC_east',
    'OAK': 'AFC_west',
    'PHI': 'NFC_east',
    'PIT': 'AFC_north',
    'SEA': 'NFC_west',
    'SF':  'NFC_west',
    'TB':  'NFC_south',
    'TEN': 'AFC_south',
    'WAS': 'NFC_east'
}

alternate_team_names = {
    'JAC': 'JAX', 'JAX': 'JAC',
    'LAC': 'SD', 'SD': 'LAC',
    'LA': 'STL', 'STL': 'LA'
}


def scrub_data(dataframe, result_to_drop=('UNK', 'tie')):

    for result in result_to_drop:
        dataframe.drop(dataframe[dataframe.result == result].index, inplace=True)

    return dataframe


def get_week_schedule(year, week, season_type='REG'):

    week_schedule = []

    schedule = nflgame.sched.games
    x = [schedule[game] for game in schedule]
    for g in x:
        if g['year'] == year and \
                g['week'] == week and \
                g['season_type'] == season_type:

            week_schedule.append(g)

    return week_schedule


def team_record(team, year, week=17, type='total'):

    if week == 0:
        return [0, 0, 0]

    if type == 'total':
        games = nflgame.games_gen(year, week=range(1, week + 1), home=team, away=team)
    elif type == 'home':
        games = nflgame.games_gen(year, week=range(1, week + 1), home=team)
    elif type == 'away':
        games = nflgame.games_gen(year, week=range(1, week + 1), away=team)

    if games is None:
        try:
            team = alternate_team_names[team]
            if type == 'total':
                games = nflgame.games_gen(year, week=range(1, week + 1), home=team, away=team)
            elif type == 'home':
                games = nflgame.games_gen(year, week=range(1, week + 1), home=team)
            elif type == 'away':
                games = nflgame.games_gen(year, week=range(1, week + 1), away=team)
        except KeyError:
            return [0, 0, 0]

    wins, losses, ties = 0, 0, 0

    try:
        for g in games:
            if g.winner == g.home + '/' + g.away:
                ties += 1
            elif g.winner == team:
                wins += 1
            else:
                losses += 1

        return [wins, losses, ties]

    except TypeError:
        return [0, 0, 0]


def team_season_win_pct(team, year, week=17, type='total'):

    if week == 0:
        return 0

    record = team_record(team, year, week, type=type)
    win, loss, tie = record
    total = sum(record)
    try:
        win_percentage = round((win + tie/2.0)/total, 3)
    except ZeroDivisionError:
        win_percentage = 0

    return win_percentage


def team_prev_season_win_pct(team, year, prev_seasons=1, type='total'):

    total_record = np.array([0, 0, 0])
    for year in range(year-1, year-prev_seasons-1, -1):
        total_record += team_record(team, year, type=type)

    win, loss, tie = total_record
    total = sum(total_record)
    try:
        win_percentage = round((win + tie*0.5)/total, 3)
    except ZeroDivisionError:
        win_percentage = 0

    return win_percentage


def divisional_flag(home, away):

    if division_dictionary[home] == division_dictionary[away]:
        return 1
    else:
        return 0


def matchup_weight(game, prev_seasons=2, verbose=False):

    # for convenience
    year, week = game['year'], game['week']
    home, away = game['home'], game['away']
    gamekey = game['gamekey']

    h_team = [home, away, home, away]
    a_team = [away, home, away, home]
    kinds = ['REG', 'REG', 'POST', 'POST']

    if verbose:
        print home, ' AT ', away, year, week
        print '---------'

    matchups = []
    for h, a, k in zip(h_team, a_team, kinds):
        try:
            game_matches = nflgame.games(year=range(year, year - (prev_seasons + 1), -1),
                                         home=h, away=a, kind=k)
            for g in game_matches:
                if g.schedule['gamekey'] != gamekey:
                    matchups.append(g)
        except (TypeError, AttributeError) as e:
            pass

    home_matchup_wins, away_matchup_wins = 0, 0
    for m in matchups:
        if verbose:
            print m, m.schedule['season_type'], m.schedule['year'], m.schedule['week']

        if m.winner == home:
            home_matchup_wins += 1
        elif m.winner == away:
            away_matchup_wins += 1
        else:
            home_matchup_wins += 0.5
            away_matchup_wins += 0.5

    total = sum([home_matchup_wins, away_matchup_wins])
    if total == 0:
        home_matchup_wpct, away_matchup_wpct = 0, 0
    else:
        home_matchup_wpct = round(home_matchup_wins / float(total), 3)
        away_matchup_wpct = round(away_matchup_wins / float(total), 3)

    if verbose:
        print '---------'
        print '({}) {} - {} ({})'.format(home_matchup_wins, home,
                                         away, away_matchup_wins)
        print '{} - {}'.format(home_matchup_wpct, away_matchup_wpct)

        print '\n'

    return home_matchup_wpct - away_matchup_wpct


def season_pt_dif_per_game(team, year):

    try:
        season_games = nflgame.games(year, home=team, away=team)
    except TypeError:
        team = alternate_team_names[team]
        season_games = nflgame.games(year, home=team, away=team)

    season_pt_dif_list = []
    for g in season_games:

        w = g.schedule['week']

        if g.is_home(team):
            team_score = g.score_home
            opponent_score = g.score_away
        else:
            team_score = g.score_away
            opponent_score = g.score_home

        point_differential = team_score - opponent_score
        season_pt_dif_list.append([w, point_differential])

    return season_pt_dif_list


def team_pt_dif_per_game_season(team, year):

    season_pt_dif_list_per_week = season_pt_dif_per_game(team, year)
    pt_dif_list = [i[1] for i in season_pt_dif_list_per_week]
    pt_dif_per_game = sum(pt_dif_list)/float(len(pt_dif_list))

    return pt_dif_per_game


def team_pt_dif_per_n_games(team, year, week):

    if week <= 1:
        return 0, 0, 0

    # season point differential so far
    season_pt_dif_list_per_week = season_pt_dif_per_game(team, year)
    week_list = [i[0] for i in season_pt_dif_list_per_week]
    try:
        last_game_week = max([w for w in week_list if w <= week-1])
        last_game_week_idx = week_list.index(last_game_week)
    except ValueError:
        return 0, 0, 0

    pt_dif_list_per_week = season_pt_dif_list_per_week[0:last_game_week_idx+1]
    pt_dif_list = [i[1] for i in pt_dif_list_per_week]
    pt_dif = sum(pt_dif_list)
    pt_dif_per_game = float(pt_dif)/len(pt_dif_list)

    # three game point differential
    if len([w for w in week_list if w <= week]) <= 3:
        three_game_pt_dif_list_per_week = season_pt_dif_list_per_week[:week-1]
    else:
        three_game_pt_dif_list_per_week = \
            season_pt_dif_list_per_week[last_game_week_idx-2:last_game_week_idx+1]

    three_game_pt_dif_list = [i[1] for i in three_game_pt_dif_list_per_week]
    three_game_pt_dif = sum(three_game_pt_dif_list)
    three_game_pt_dif_per_game = float(three_game_pt_dif)/len(three_game_pt_dif_list)

    # five game point differential
    if len([w for w in week_list if w <= week]) <= 5:
        five_game_pt_dif_list_per_week = season_pt_dif_list_per_week[:week-1]

    else:
        five_game_pt_dif_list_per_week = \
            season_pt_dif_list_per_week[last_game_week_idx-4:last_game_week_idx+1]

    five_game_pt_dif_list = [i[1] for i in five_game_pt_dif_list_per_week]

    five_game_pt_dif = sum(five_game_pt_dif_list)
    five_game_pt_dif_per_game = float(five_game_pt_dif)/len(five_game_pt_dif_list)

    return \
        round(pt_dif_per_game, 3), \
        round(three_game_pt_dif_per_game, 3), \
        round(five_game_pt_dif_per_game, 3)


def turnovers_per_week(team, year):

    try:
        season_games = nflgame.games(year, home=team, away=team)
    except TypeError:
        team = alternate_team_names[team]
        season_games = nflgame.games(year, home=team, away=team)

    week_list = []
    turnover_list = []
    turnover_dif_list = []

    for g in season_games:
        w = g.schedule['week']
        if g.is_home(team):
            team_turnovers = g.stats_home.turnovers
            opponent_turnovers = g.stats_away.turnovers
            turnover_dif = opponent_turnovers - team_turnovers
        else:
            team_turnovers = g.stats_away.turnovers
            opponent_turnovers = g.stats_home.turnovers
            turnover_dif = opponent_turnovers - team_turnovers

        week_list.append(w)
        turnover_list.append(team_turnovers)
        turnover_dif_list.append(turnover_dif)

    turnover_dict = {'week': week_list,
                     'turnovers': turnover_list,
                     'turnover_dif': turnover_dif_list}

    return turnover_dict


def turnovers_per_game_season(team, year):

    season_turnover_dict = turnovers_per_week(team, year)
    turnovers = season_turnover_dict['turnovers']
    turnover_dif = season_turnover_dict['turnover_dif']

    turnovers_per_game = np.mean(turnovers)
    turnover_dif_per_game = np.mean(turnover_dif)

    return {'turnovers_per_game': turnovers_per_game,
            'turnover_dif_per_game': turnover_dif_per_game}


def turnovers_per_game(team, year, week):

    empty_dict = {
        'season_turnovers_per_game': 0,
        '3game_turnovers_per_game': 0,
        '5game_turnovers_per_game': 0,
        'season_turnover_dif_per_game': 0,
        '3game_turnover_dif_per_game': 0,
        '5game_turnover_dif_per_game': 0
    }

    if week <= 1:
        return empty_dict

    # season turnovers so far
    season_turnover_dict = turnovers_per_week(team, year)
    week_list = season_turnover_dict['week']
    turnovers_list = season_turnover_dict['turnovers']
    turnover_dif_list = season_turnover_dict['turnover_dif']

    try:
        last_game_week = max([w for w in week_list if w <= week-1])
        last_game_week_idx = week_list.index(last_game_week)
    except ValueError:
        return empty_dict

    # Season so far
    turnovers_per_game_list = turnovers_list[0:last_game_week_idx+1]
    season_turnovers_per_game = np.mean(turnovers_per_game_list)
    turnover_dif_per_game_list = turnover_dif_list[0:last_game_week_idx+1]
    season_turnover_dif_per_game = np.mean(turnover_dif_per_game_list)

    # Last three game turnovers
    if len(turnovers_per_game_list) <= 3:
        team_3game_turnovers_per_game = season_turnovers_per_game
        team_3game_turnover_dif_per_game = season_turnover_dif_per_game
    else:
        team_3game_turnovers_per_game = np.mean(turnovers_per_game_list[-3:])
        team_3game_turnover_dif_per_game = np.mean(turnover_dif_per_game_list[-3:])

    # Last five game turnovers
    if len(turnovers_per_game_list) <= 5:
        team_5game_turnovers_per_game = season_turnovers_per_game
        team_5game_turnover_dif_per_game = season_turnover_dif_per_game
    else:
        team_5game_turnovers_per_game = np.mean(turnovers_per_game_list[-5:])
        team_5game_turnover_dif_per_game = np.mean(turnover_dif_per_game_list[-5:])

    turnover_dict = {
        'season_turnovers_per_game': season_turnovers_per_game,
        '3game_turnovers_per_game': team_3game_turnovers_per_game,
        '5game_turnovers_per_game': team_5game_turnovers_per_game,
        'season_turnover_dif_per_game': season_turnover_dif_per_game,
        '3game_turnover_dif_per_game': team_3game_turnover_dif_per_game,
        '5game_turnover_dif_per_game': team_5game_turnover_dif_per_game
    }

    # Rounding
    for key in turnover_dict:
        turnover_dict[key] = round(turnover_dict[key], 3)

    return turnover_dict


def third_down_pct_game(team, game):

    attempts, conversions = 0, 0
    play_generator = game.drives.plays()

    for play in play_generator.filter(team=team, third_down_att=1):
        attempts += 1.
        conversions += play.third_down_conv

    if conversions == 0:
        third_down_pct = 0
    else:
        third_down_pct = round(conversions/attempts, 3)

    return third_down_pct


def third_down_pct_per_week(team, year):

    try:
        season_games = nflgame.games(year, home=team, away=team)
    except TypeError:
        team = alternate_team_names[team]
        season_games = nflgame.games(year, home=team, away=team)

    week_list = []
    third_down_pct_list = []

    for g in season_games:
        week_list.append(g.schedule['week'])
        third_down_pct_list.append(third_down_pct_game(team, g))

    return {'week': week_list,
            'third_down_pct': third_down_pct_list}


def third_down_pct_per_game(team, year, week):

    empty_dict = {
        'season_3down_pct_per_game': 0,
        '3game_3down_pct_per_game': 0,
        '5game_3down_pct_per_game': 0}

    if week <= 1:
        return empty_dict

    # season third down percentage so far
    season_tdp_dict = third_down_pct_per_week(team, year)
    week_list = season_tdp_dict['week']
    tdp_list = season_tdp_dict['third_down_pct']

    try:
        last_game_week = max([w for w in week_list if w <= week-1])
        last_game_week_idx = week_list.index(last_game_week)
    except ValueError:
        return empty_dict

    # Season so far
    tdp_per_game_list = tdp_list[0:last_game_week_idx+1]
    season_tdp_per_game = np.mean(tdp_per_game_list)

    # Last three games
    if len(tdp_per_game_list) <= 3:
        team_3game_tdp = season_tdp_per_game
    else:
        team_3game_tdp = np.mean(tdp_per_game_list[-3:])

    # Last five games
    if len(tdp_per_game_list) <= 5:
        team_5game_tdp = season_tdp_per_game
    else:
        team_5game_tdp = np.mean(tdp_per_game_list[-5:])

    tdp_dict = {
        'season_3down_pct_per_game': season_tdp_per_game,
        '3game_3down_pct_per_game': team_3game_tdp,
        '5game_3down_pct_per_game': team_5game_tdp}

    # Rounding
    for key in tdp_dict:
        tdp_dict[key] = round(tdp_dict[key], 3)

    return tdp_dict


if __name__ == "__main__":

    test_team = 'TB'
    year = 2016



    data = []
    for team in nflgame.teams:

        data.append([team[0],
                     third_down_pct_per_game(team[0], year, week=17)['season_3down_pct_per_game'],
                     team_record(team[0], year)])

    df = pd.DataFrame(data, columns=['team', 'tdp', 'record'])
    print df.sort_values('tdp', ascending=False)
