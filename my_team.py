# FILE: src/contest/agents/myTeam.py


import random, time

import contest.game as game

import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point



#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='HybridReflexAgent1', second='HybridReflexAgent3', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.
    """
    return [eval(first)(first_index), eval(second)(second_index)]

##########
# Agents #
##########

class HybridReflexAgent1(CaptureAgent):  #RedOne, more offensive one
    """
    A reflex agent that can switch between offensive and defensive modes.
    """
    FOOD_THRESHOLD = 6  #threshold for the number of food points before returning to base

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        self.food_collected = 0  #initialize food collected counter
        CaptureAgent.register_initial_state(self, game_state)
    
    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        #get companion positions
        companions = [game_state.get_agent_state(i) for i in self.get_team(game_state) if i != self.index]
        companion_positions = [a.get_position() for a in companions if a.get_position() is not None]

        actions = game_state.get_legal_actions(self.index)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        #evaluate all actions and choose the best one
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        
        #filter positions were other companions are
        filtered_actions = []
        for action in best_actions:
            successor = self.get_successor(game_state, action)
            next_position = successor.get_agent_position(self.index)
            if next_position not in companion_positions:  #avoid positions already taken by companions
                filtered_actions.append(action)

        #use filtered actions if available, otherwise default to best_actions
        best_actions = filtered_actions if filtered_actions else best_actions

        #check if there is only one possible move (not taking into account stop)
        if len(game_state.get_legal_actions(self.index)) == 2:
            actions = [action for action in game_state.get_legal_actions(self.index) if action != Directions.STOP]
            if actions:
                successor = self.get_successor(game_state, actions[0])
                next_position = successor.get_agent_position(self.index)

                #if the next position has food, increment the counter
                if next_position in self.get_food(game_state).as_list():
                    self.food_collected += 1
                return actions[0]
        
        #if there are visible invaders, move toward the closest one
        if len(invaders) > 0:
            invader_positions = [a.get_position() for a in invaders]
            best_dist = float('inf')
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                #compute distance to closest invader
                dist = min(self.get_maze_distance(pos2, inv) for inv in invader_positions)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            if best_action is not None:
                successor = self.get_successor(game_state, best_action)
                next_position = successor.get_agent_position(self.index)

                # if the next position has food, increment the counter
                if next_position in self.get_food(game_state).as_list():
                    self.food_collected += 1
                return best_action

        agent_state = game_state.get_agent_state(self.index)
        #if there are only a few food left or the threshold is met, head back to start 
        food_left = len(self.get_food(game_state).as_list())
        if food_left <= 2 or self.food_collected >= self.FOOD_THRESHOLD:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            if self.food_collected >= self.FOOD_THRESHOLD and not agent_state.is_pacman: #make sure it is not in pacman form (hasn't come back)
                self.food_collected = 0  #reset food collected counter after returning to base
            successor = self.get_successor(game_state, best_action)
            next_position = successor.get_agent_position(self.index)

            #if the next position has food, increment the counter
            if next_position in self.get_food(game_state).as_list():
                self.food_collected += 1
            return best_action
        #select a random best choice
        best_action = random.choice(best_actions)

        #if the next position has food, increment the counter
        successor = self.get_successor(game_state, best_action)
        next_position = successor.get_agent_position(self.index)
        if next_position in self.get_food(game_state).as_list():
            self.food_collected += 1


        return best_action


    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != util.nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = successor.get_agent_state(self.index).get_position()
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)

        #compute distance to the nearest food
        food_list = self.get_food(successor).as_list()
        if len(food_list) > 0:  #this should always be True,  but better safe than sorry
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        #compute distance to the nearest ghost
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        if len(ghosts) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in ghosts]
            features['distance_to_ghost'] = min(dists)

        #compute distance to invaders we can see
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        features['on_defense'] = 0
        #if there are invaders seen
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)
            features['on_defense'] = 1 # put it on defense mode

        teammate_index = 2  #assuming 2 agents total (we don't know if the index will change, maybe it doesn't work in some cases but is not something super important for our aproach)
        teammate_pos = game_state.get_agent_position(teammate_index)
        if teammate_pos is not None:
        #check if both agents are either Pacman or Ghost, we don't care if they are different surely they are already apart
            my_state = game_state.get_agent_state(self.index)
            teammate_state = game_state.get_agent_state(teammate_index)
            
            both_pacman = my_state.is_pacman and teammate_state.is_pacman
            both_ghost = not my_state.is_pacman and not teammate_state.is_pacman
            
            if both_pacman or both_ghost:
                distance_to_teammate = self.get_maze_distance(my_pos, teammate_pos)
                if distance_to_teammate == 0:
                    features['distance_to_teammate'] = 3
                else:
                    features['distance_to_teammate'] = 1/distance_to_teammate  #smaller distance bigger importance
            else:
                features['distance_to_teammate'] = 0


        #take into account if it does stop or reverse action (sometimes we don't want this)
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1


        if len(ghosts) > 0:
            ghost_distances = [self.get_maze_distance(my_pos, ghost.get_position()) for ghost in ghosts]
            min_ghost_distance = min(ghost_distances)
            if min_ghost_distance <= 1.5:  #dodge behavior for close ghosts
                features['ghost_danger'] = 1 / (min_ghost_distance + 1)  #higher weight for closer ghosts
            else:
                features['ghost_danger'] = 0
        else:
            features['distance_to_ghost'] = 0
            features['ghost_danger'] = 0
        

        #check for dead_ends, our function
        if self.is_dead_end(game_state, 4):
            features['dead_end'] = 1
        else:
            features['dead_end'] = 0
        
        scared_enemies = [a for a in enemies if a.scared_timer > 0 and a.get_position() is not None]

        #if there are any scared enemies, prioritize them
        if len(scared_enemies) > 0:
            scared_positions = [a.get_position() for a in scared_enemies]
            
            #compute distance to the nearest scared enemy
            min_dist = min([self.get_maze_distance(game_state.get_agent_position(self.index), pos) for pos in scared_positions])
            features['distance_to_scared_enemy'] = min_dist
            
            #prioritize hunting if a scared enemy is within range
            if min_dist <= 5:
                features['hunt_scared_enemy'] = 1
            else:
                features['hunt_scared_enemy'] = 0
        else:
            features['hunt_scared_enemy'] = 0 

        return features
    
    def is_dead_end(self, game_state, depth=4): #we are not sure why it doesn't work, we don't want to take it out so it doesnt mess the code
        """
        Detects if a given position leads to a dead end within a certain depth.
        """
        position = game_state.get_agent_position(self.index)
        queue = [(position, 0)]  # (current position, depth level)
        visited = set()
        dead_end = True

        while queue:
            pos, level = queue.pop(0)
            if level >= depth:  #safe zone after depth
                dead_end = False
                break

            visited.add(pos)
            legal_actions = game_state.get_legal_actions(self.index)
            next_positions = [game_state.generate_successor(self.index, action).get_agent_position(self.index) 
                            for action in legal_actions if action != Directions.STOP]

            for next_pos in next_positions:
                if next_pos not in visited:
                    queue.append((next_pos, level + 1))
        return dead_end
    
    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the state or action
        """
        successor = self.get_successor(game_state, action)
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        scared_enemies = [a for a in enemies if a.scared_timer > 0 and a.get_position() is not None]
        successor_score = 0
        distance_to_food = 0
        distance_to_ghost = 0
        num_invaders = 0
        on_defense = 0
        invader_distance = 0
        stop = 0
        reverse = 0
        hunt_scared_enemy = 0
        if len(invaders)>0:  #weights if there are invaders
            num_invaders = -1000
            on_defense = -1000
            invader_distance = -30
            stop = -100
            reverse = -2
        elif len(scared_enemies) > 0: #weight if there are scared ghosts
            hunt_scared_enemy = 100  
            distance_to_food = -3
            successor_score = 100
            distance_to_ghost = -4
        else:  #normal attack mode
            distance_to_food = -3
            successor_score = 100
            distance_to_ghost = 4


        return {
            'successor_score': successor_score,
            'distance_to_food': distance_to_food,
            'distance_to_ghost': distance_to_ghost,
            'num_invaders': num_invaders,
            'on_defense': on_defense,
            'invader_distance': invader_distance,
            'stop': stop,
            'reverse': reverse,
            'distance_to_teammate': -1,
            'dead_end' : -100,
            'hunt_scared_enemy' : hunt_scared_enemy
        }
    
    
    
class HybridReflexAgent3(CaptureAgent): #Other pacman
    """
    A reflex agent that can switch between offensive and defensive modes.
    """
    FOOD_THRESHOLD = 2  #threshold for the number of food points before returning to base

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        self.food_collected = 0  #initialize food collected counter
        CaptureAgent.register_initial_state(self, game_state)
    
    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        #get companion positions
        companions = [game_state.get_agent_state(i) for i in self.get_team(game_state) if i != self.index]
        companion_positions = [a.get_position() for a in companions if a.get_position() is not None]
        actions = game_state.get_legal_actions(self.index)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        #evaluate all actions and choose the best one
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        
        #filter positions were other companions are
        filtered_actions = []
        for action in best_actions:
            successor = self.get_successor(game_state, action)
            next_position = successor.get_agent_position(self.index)
            if next_position not in companion_positions:  #avoid positions already taken by companions
                filtered_actions.append(action)

        #use filtered actions if available, otherwise default to best_actions
        best_actions = filtered_actions if filtered_actions else best_actions

        #check if there is only one possible move (not taking into account stop)
        if len(game_state.get_legal_actions(self.index)) == 2:
            actions = [action for action in game_state.get_legal_actions(self.index) if action != Directions.STOP]
            if actions:
                successor = self.get_successor(game_state, actions[0])
                next_position = successor.get_agent_position(self.index)

                #if the next position has food, increment the counter
                if next_position in self.get_food(game_state).as_list():
                    self.food_collected += 1
                return actions[0]
        
        #if there are visible invaders, move toward the closest one
        if len(invaders) > 0:
            invader_positions = [a.get_position() for a in invaders]
            best_dist = float('inf')
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                #compute distance to closest invader
                dist = min(self.get_maze_distance(pos2, inv) for inv in invader_positions)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            if best_action is not None:
                successor = self.get_successor(game_state, best_action)
                next_position = successor.get_agent_position(self.index)

                #if the next position has food, increment the counter
                if next_position in self.get_food(game_state).as_list():
                    self.food_collected += 1
                return best_action
        
        agent_state = game_state.get_agent_state(self.index)
        
        #if there are only a few food left, head back to start
        food_left = len(self.get_food(game_state).as_list())
        if food_left <= 2 or self.food_collected >= self.FOOD_THRESHOLD:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            if self.food_collected >= self.FOOD_THRESHOLD and not agent_state.is_pacman:
                self.food_collected = 0  #reset food collected counter after returning to base
            successor = self.get_successor(game_state, best_action)
            next_position = successor.get_agent_position(self.index)

            #if the next position has food, increment the counter
            if next_position in self.get_food(game_state).as_list():
                self.food_collected += 1
            return best_action
        best_action = random.choice(best_actions)
        successor = self.get_successor(game_state, best_action)
        next_position = successor.get_agent_position(self.index)

        #if the next position has food, increment the counter
        if next_position in self.get_food(game_state).as_list():
            self.food_collected += 1


        return best_action


    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != util.nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = successor.get_agent_state(self.index).get_position()
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)
        #compute distance to the nearest food
        food_list = self.get_food(successor).as_list()
        if len(food_list) > 0:  #this should always be True,  but better safe than sorry
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        #compute distance to the nearest ghost
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        if len(ghosts) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in ghosts]
            features['distance_to_ghost'] = min(dists)

        #compute distance to invaders we can see
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        teammate_index = 2  #assuming 2 agents total (we don't know if the index will change, maybe it doesn't work in some cases but is not something super important for our aproach)
        teammate_pos = game_state.get_agent_position(teammate_index)
        if teammate_pos is not None:
        #check if both agents are either Pacman or Ghost, we don't care if they are different surely they are already apart
            my_state = game_state.get_agent_state(self.index)
            teammate_state = game_state.get_agent_state(teammate_index)
            
            both_pacman = my_state.is_pacman and teammate_state.is_pacman
            both_ghost = not my_state.is_pacman and not teammate_state.is_pacman
            
            if both_pacman or both_ghost:
                distance_to_teammate = self.get_maze_distance(my_pos, teammate_pos)
                if distance_to_teammate == 0:
                    features['distance_to_teammate'] = 3
                else:
                    features['distance_to_teammate'] = 1/distance_to_teammate #smaller distance bigger importance
            else:
                features['distance_to_teammate'] = 0

        #take into account if it does stop or reverse action (sometimes we don't want this)
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        if len(ghosts) > 0:
            ghost_distances = [self.get_maze_distance(my_pos, ghost.get_position()) for ghost in ghosts]
            min_ghost_distance = min(ghost_distances)
            if min_ghost_distance < 1:
                features['distance_to_ghost'] = min_ghost_distance
            if min_ghost_distance <= 3:  #dodge behavior for close ghosts
                features['ghost_danger'] = 1 / (min_ghost_distance + 1)  #higher weight for closer ghosts
            else:
                features['ghost_danger'] = 0
        else:
            features['distance_to_ghost'] = 0
            features['ghost_danger'] = 0

        #check for dead_ends, our function
        if self.is_dead_end(game_state, 4):
            features['dead_end'] = 1
        else:
            features['dead_end'] = 0
        
        scared_enemies = [a for a in enemies if a.scared_timer > 0 and a.get_position() is not None]

        #if there are any scared enemies, prioritize them
        if len(scared_enemies) > 0:

            scared_positions = [a.get_position() for a in scared_enemies]
            
            #compute distance to the nearest scared enemy
            min_dist = min([self.get_maze_distance(game_state.get_agent_position(self.index), pos) for pos in scared_positions])
            features['distance_to_scared_enemy'] = min_dist
            
            #prioritize hunting if a scared enemy is within range
            if min_dist <= 5:  
                features['hunt_scared_enemy'] = 1
            else:
                features['hunt_scared_enemy'] = 0
        else:
            features['hunt_scared_enemy'] = 0  


        return features
    
    def is_dead_end(self, game_state, depth=4):
        """
        Detects if a given position leads to a dead end within a certain depth.
        """
        position = game_state.get_agent_position(self.index)
        queue = [(position, 0)]  # (current position, depth level)
        visited = set()
        dead_end = True

        while queue:
            pos, level = queue.pop(0)
            if level >= depth:  #safe zone after depth
                dead_end = False
                break

            visited.add(pos)
            legal_actions = game_state.get_legal_actions(self.index)
            next_positions = [game_state.generate_successor(self.index, action).get_agent_position(self.index) 
                            for action in legal_actions if action != Directions.STOP]

            for next_pos in next_positions:
                if next_pos not in visited:
                    queue.append((next_pos, level + 1))
        
        return dead_end
    
    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the state or action
        """
        food_left = len(self.get_food(game_state).as_list())
        successor = self.get_successor(game_state, action)
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        scared_enemies = [a for a in enemies if a.scared_timer > 0 and a.get_position() is not None]
        successor_score = 0
        distance_to_food = 0
        distance_to_ghost = 0
        num_invaders = 0
        on_defense = 0
        invader_distance = 0
        stop = 0
        reverse = 0
        hunt_scared_enemy = 0
        if len(invaders)>0:  #different weights on diff situations, enemies found, scared enemies found, standar attack
            num_invaders = -1000
            on_defense = -1000
            invader_distance = -30
            stop = -100
            reverse = -2
        elif len(scared_enemies) > 0:
            hunt_scared_enemy = 1000  
            distance_to_food = -3
            successor_score = 100
            distance_to_ghost = -6
        #elif food_left < 20:   we don't know why but this gives an error, we wanted to put it in "defense mode" if the condition was met
            #print("ondefenseduty")
            #num_invaders = -1000, 
            #on_defense = -100, 
            #invader_distance = -10, 
            #stop = -100, 
            #reverse = -2
        else:
            distance_to_food = -3
            successor_score = 100
            distance_to_ghost = 4
        


        return {
            'successor_score': successor_score,
            'distance_to_food': distance_to_food,
            'distance_to_ghost': distance_to_ghost,
            'num_invaders': num_invaders,
            'on_defense': on_defense,
            'invader_distance': invader_distance,
            'stop': stop,
            'reverse': reverse,
            'distance_to_teammate': -1,
            'dead_end' : -100,
            'hunt_scared_enemy' : hunt_scared_enemy
        }
    
class DefensiveReflexAgent(CaptureAgent): #not used, but it was a try
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
    FOOD_THRESHOLD = 2
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.food_collected = 0  #initialize food collected counter
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        companions = [game_state.get_agent_state(i) for i in self.get_team(game_state) if i != self.index]
        companion_positions = [a.get_position() for a in companions if a.get_position() is not None]
        actions = game_state.get_legal_actions(self.index)
        actions = game_state.get_legal_actions(self.index)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        filtered_actions = []
        for action in best_actions:
            successor = self.get_successor(game_state, action)
            next_position = successor.get_agent_position(self.index)
            if next_position not in companion_positions: 
                filtered_actions.append(action)


        best_actions = filtered_actions if filtered_actions else best_actions

        if len(invaders) > 0:
            invader_positions = [a.get_position() for a in invaders]
            best_dist = float('inf')
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)

                dist = min(self.get_maze_distance(pos2, inv) for inv in invader_positions)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            if best_action is not None:
                return best_action
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]

        food_left = len(self.get_food(game_state).as_list())
        ghost_distance_threshold = 5
        if ghosts:
            ghost_distances = [
                self.get_maze_distance(game_state.get_agent_position(self.index), ghost.get_position())
                for ghost in ghosts
            ]
            nearest_ghost_distance = min(ghost_distances)
        else:
            nearest_ghost_distance = float('inf')
        agent_state = game_state.get_agent_state(self.index)
        if food_left <= 2 or self.food_collected >= self.FOOD_THRESHOLD or (len(ghosts) > 0 and nearest_ghost_distance <= ghost_distance_threshold) and not agent_state.is_pacman:
            print("run away")
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            if self.food_collected >= self.FOOD_THRESHOLD:
                self.food_collected = 0  
            return best_action
        
        return random.choice(best_actions)

    def is_dead_end(self, game_state, depth=4):
        """
        Detects if a given position leads to a dead end within a certain depth.
        """
        position = game_state.get_agent_position(self.index)
        queue = [(position, 0)]
        visited = set()
        dead_end = True

        while queue:
            pos, level = queue.pop(0)
            if level >= depth:  
                dead_end = False
                break

            visited.add(pos)
            legal_actions = game_state.get_legal_actions(self.index)
            next_positions = [game_state.generate_successor(self.index, action).get_agent_position(self.index) 
                            for action in legal_actions if action != Directions.STOP]

            for next_pos in next_positions:
                if next_pos not in visited:
                    queue.append((next_pos, level + 1))
        
        return dead_end
    
    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights


    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = successor.get_agent_state(self.index).get_position()
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)
        # Compute distance to the nearest food
        food_list = self.get_food(successor).as_list()
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        # Compute distance to the nearest ghost
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        if len(ghosts) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in ghosts]
            features['distance_to_ghost'] = min(dists)

        # Compute distance to invaders we can see
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        teammate_index = 2  # Assuming 2 agents total
        teammate_pos = game_state.get_agent_position(teammate_index)
        if teammate_pos is not None:
        # Check if both agents are either Pacman or Ghost
            my_state = game_state.get_agent_state(self.index)
            teammate_state = game_state.get_agent_state(teammate_index)
            
            both_pacman = my_state.is_pacman and teammate_state.is_pacman
            both_ghost = not my_state.is_pacman and not teammate_state.is_pacman
            
            if both_pacman or both_ghost:
                distance_to_teammate = self.get_maze_distance(my_pos, teammate_pos)
                if distance_to_teammate == 0:
                    features['distance_to_teammate'] = 3
                else:
                    features['distance_to_teammate'] = 1/distance_to_teammate
            else:
                features['distance_to_teammate'] = 0

        # Determine if the agent should be on defense
        features['on_defense'] = 0
        if len(invaders) > 0:
            features['on_defense'] = 1

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        if len(ghosts) > 0:
            ghost_distances = [self.get_maze_distance(my_pos, ghost.get_position()) for ghost in ghosts]
            min_ghost_distance = min(ghost_distances)
            if min_ghost_distance < 1:
                features['distance_to_ghost'] = min_ghost_distance
            if min_ghost_distance <= 3:  # Dodge behavior for close ghosts
                features['ghost_danger'] = 1 / (min_ghost_distance + 1)  # Higher weight for closer ghosts
            else:
                features['ghost_danger'] = 0
        else:
            features['distance_to_ghost'] = 0
            features['ghost_danger'] = 0
        
        if self.is_dead_end(game_state, 4):
            features['dead_end'] = 1
        else:
            features['dead_end'] = 0
        
        scared_enemies = [a for a in enemies if a.scared_timer > 0 and a.get_position() is not None]

        # If there are any scared enemies, prioritize them
        if len(scared_enemies) > 0:
            # Get the position of the scared enemies
            scared_positions = [a.get_position() for a in scared_enemies]
            
            # Compute distance to the nearest scared enemy
            min_dist = min([self.get_maze_distance(game_state.get_agent_position(self.index), pos) for pos in scared_positions])
            features['distance_to_scared_enemy'] = min_dist
            
            # Prioritize hunting if a scared enemy is within range
            if min_dist <= 5:  # For example, hunting only if within range of 5
                features['hunt_scared_enemy'] = 1
            else:
                features['hunt_scared_enemy'] = 0
        else:
            features['hunt_scared_enemy'] = 0  # No enemies to hunt
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()


        

        return features



    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the state or action
        """
        food_left = len(self.get_food(game_state).as_list())
        successor = self.get_successor(game_state, action)
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        scared_enemies = [a for a in enemies if a.scared_timer > 0 and a.get_position() is not None]
        successor_score = 0
        distance_to_food = 0
        distance_to_ghost = 0
        num_invaders = 0
        on_defense = 0
        invader_distance = 0
        stop = 0
        reverse = 0
        ghostdanger = 0
        hunt_scared_enemy = 0
        if len(invaders)>0:
            print("enemies found")
            num_invaders = -1000
            on_defense = -1000
            invader_distance = -30
            stop = -100
            reverse = -2
        elif len(scared_enemies) > 0:
            hunt_scared_enemy = 100  
            distance_to_food = -3
            successor_score = 100
            distance_to_ghost = -4
        elif food_left < 18:
            num_invaders = -1000, 
            on_defense = 100, 
            invader_distance = -10, 
            stop = -100, 
            reverse = -2
        else:
            distance_to_food = -0.5
            successor_score = 10
            distance_to_ghost = 4
            ghostdanger = 2


        return {
            'successor_score': successor_score,
            'distance_to_food': distance_to_food,
            'distance_to_ghost': distance_to_ghost,
            'num_invaders': num_invaders,
            'on_defense': on_defense,
            'invader_distance': invader_distance,
            'stop': stop,
            'reverse': reverse,
            'distance_to_teammate': 0,
            'ghost_danger': ghostdanger,
            'hunt_scared_enemy' : hunt_scared_enemy
        }


        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}