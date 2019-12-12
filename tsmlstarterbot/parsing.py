import numpy as np
import pandas as pd
import pickle

from tsmlstarterbot.common import *


def angle(x, y):
    radians = math.atan2(y, x)
    if radians < 0:
        radians = radians + 2 * math.pi
    return round(radians / math.pi * 180)


def find_winner(data):
    for player, stats in data['stats'].items():
        if stats['rank'] == 1:
            return player
    return -1


def angle_dist(a1, a2):
    return (a1 - a2 + 360) % 360


def find_target_planet(bot_id, current_frame, planets, move):
    """
    Find a planet which the ship tried to go to. We try to find it by looking at the angle that the ship moved
    with and the angle between the ship and the planet.
    :param bot_id: id of bot to imitate
    :param current_frame: current frame
    :param planets: planets data
    :param move: current move to analyze
    :return: id of the planet that ship was moving towards
    """

    if move['type'] == 'dock':
        # If the move was to dock, we know the planet we wanted to move towards
        return move['planet_id']
    if move['type'] != 'thrust':
        # If the move was not "thrust" (i.e. it was "undock"), there is no angle to analyze
        return -1

    ship_angle = move['angle']
    ship_data = current_frame['ships'][bot_id][str(move['shipId'])]
    ship_x = ship_data['x']
    ship_y = ship_data['y']

    optimal_planet = -1
    optimal_angle = -1
    for planet_data in planets:
        planet_id = str(planet_data['id'])
        if planet_id not in current_frame['planets'] or current_frame['planets'][planet_id]['health'] <= 0:
            continue

        planet_x = planet_data['x']
        planet_y = planet_data['y']
        a = angle(planet_x - ship_x, planet_y - ship_y)
        # We try to find the planet with minimal angle distance
        if optimal_planet == -1 or angle_dist(ship_angle, a) < angle_dist(ship_angle, optimal_angle):
            optimal_planet = planet_id
            optimal_angle = a

    return optimal_planet


def format_data_for_training(data):
    """
    Create numpy array with planet features ready to feed to the neural net.
    :param data: parsed features
    :return: numpy array of shape (number of frames, MAP_MAX_HEIGHT//SCALE_FACTOR, MAP_MAX_WIDTH//SCALE_FACTOR, 4)
    """
    training_input = []
    training_output = []
    for d in data:
        features, expected_output = d

        if len(expected_output.values()) == 0:
            continue
        fm = np.array(features)


        output = [0] * PLANET_MAX_NUM
        for planet_id, p in expected_output.items():
            output[int(planet_id)] = p
        result = np.array(output)

        training_input.append(fm)
        training_output.append(result)

    return np.array(training_input), np.array(training_output)

def parse(all_games_json_data, bot_to_imitate=None, dump_features_location=None):
    """
    Parse the games to compute features. This method creates our image as an input.

    :param all_games_json_data: list of json dictionaries describing games
    :param bot_to_imitate: name of the bot to imitate or None if we want to imitate the bot who won the most games
    :param dump_features_location: location where to serialize the features
    :return: data ready for training
    """
    print("Parsing data...")

    parsed_games = 0

    training_data = []

    if bot_to_imitate is None:
        print("No bot name provided, choosing the bot with the highest number of games won...")
        players_games_count = {}
        for json_data in all_games_json_data:
            w = find_winner(json_data)
            p = json_data['player_names'][int(w)]
            if p not in players_games_count:
                players_games_count[p] = 0
            players_games_count[p] += 1

        bot_to_imitate = max(players_games_count, key=players_games_count.get)
    print("Bot to imitate: {}.".format(bot_to_imitate))
    games_count = 0
    for json_data in all_games_json_data:
        games_count += 1
        if games_count % 25 == 0:
            print("Games processed: {}.".format(games_count))
        frames = json_data['frames']
        moves = json_data['moves']
        width = json_data['width']
        height = json_data['height']

        # For each game see if bot_to_imitate played in it
        if bot_to_imitate not in set(json_data['player_names']):
            continue
        # We train on all the games of the bot regardless whether it won or not.
        bot_to_imitate_id = str(json_data['player_names'].index(bot_to_imitate))

        parsed_games = parsed_games + 1
        game_training_data = []

        # Ignore the last frame, no decision to be made there
        for idx in range(len(frames) - 1):

            current_moves = moves[idx]
            current_frame = frames[idx]

            if bot_to_imitate_id not in current_frame['ships'] or len(current_frame['ships'][bot_to_imitate_id]) == 0:
                continue

            current_planets = current_frame['planets']

            # find % allocation for all ships
            all_moving_ships = 0
            allocations = {}

            # for each planet we want to find how many ships are being moved towards it now
            for ship_id, ship_data in current_frame['ships'][bot_to_imitate_id].items():
                if ship_id in current_moves[bot_to_imitate_id][0]:
                    p = find_target_planet(bot_to_imitate_id, current_frame,
                                           json_data['planets'],
                                           current_moves[bot_to_imitate_id][0][ship_id],
                                           )
                    planet_id = int(p)
                    if planet_id < 0 or planet_id >= PLANET_MAX_NUM:
                        continue

                    if p not in allocations:
                        allocations[p] = 0
                    allocations[p] = allocations[p] + 1
                    all_moving_ships = all_moving_ships + 1

            if all_moving_ships == 0:
                continue

            # Compute what % of the ships should be sent to given planet
            for planet_id, allocated_ships in allocations.items():
                allocations[planet_id] = allocated_ships / all_moving_ships

            # Compute features
            for planet_id in range(PLANET_MAX_NUM):

                if str(planet_id) not in current_planets:
                    continue
                planet_data = current_planets[str(planet_id)]
                planet_base_data = json_data['planets'][planet_id]

                ownership = 1
                if str(planet_data['owner']) == bot_to_imitate_id:
                    ownership = 2
                elif planet_data['owner'] is not None:
                    ownership = 3

                remaining_docking_spots = planet_base_data['docking_spots'] - len(planet_data['docked_ships'])
                feature_matrix = [[[0 for _ in range(NUM_IMAGE_LAYERS)] for _ in range(MAP_MAX_HEIGHT//SCALE_FACTOR)] for _ in range(MAP_MAX_WIDTH//SCALE_FACTOR)]
                
                x = int(planet_base_data['x']//SCALE_FACTOR)
                y = int(planet_base_data['y']//SCALE_FACTOR)
                radius = int(planet_base_data['r']//SCALE_FACTOR)
                for i in range(radius):
                    for j in range(radius):
                        if i**2 + j**2 <= radius**2 and x > i and y > j and x+i < MAP_MAX_WIDTH//SCALE_FACTOR and y+j < MAP_MAX_HEIGHT//SCALE_FACTOR:
                            feature_matrix[x+i][y+j][0] = ownership
                            feature_matrix[x-i][y+j][0] = ownership
                            feature_matrix[x+i][y-j][0] = ownership
                            feature_matrix[x-i][y-j][0] = ownership
                            if ownership != 3:
                                feature_matrix[x+i][y+j][3] = remaining_docking_spots
                                feature_matrix[x-i][y+j][3] = remaining_docking_spots
                                feature_matrix[x+i][y-j][3] = remaining_docking_spots
                                feature_matrix[x-i][y-j][3] = remaining_docking_spots
            for owner_id, ship_list in current_frame['ships'].items():
                for ship_id, ship_data in ship_list.items():
                    if owner_id == bot_to_imitate_id:
                        feature_matrix[int(ship_data['x']/SCALE_FACTOR)][int(ship_data['y']/SCALE_FACTOR)][1] += 1
                    else: 
                        feature_matrix[int(ship_data['x']/SCALE_FACTOR)][int(ship_data['y']/SCALE_FACTOR)][2] += 1
            game_training_data.append((feature_matrix, allocations))
        training_data.append(game_training_data)

    if parsed_games == 0:
        raise Exception("Didn't find any matching games. Try different bot.")

    if dump_features_location is not None:
        serialize_data(training_data, dump_features_location)

    flat_training_data = [item for sublist in training_data for item in sublist]
    print("Data parsed, parsed {} games, total frames: {}".format(parsed_games, len(flat_training_data)))

    return format_data_for_training(flat_training_data)