import numpy as np

def attackScore(ball_pos, blue_team):
    score = 0
    for player in blue_team:
        dist = np.linalg.norm(ball_pos - player)
        score += 100 / (dist+1)
    return score

def dodgeScore(ball_pos, blue_team):
    score = 0
    for player in blue_team:
        dist = np.linalg.norm(ball_pos - player)
        score += dist**2
    return score

def teamSpacingScore(blue_team, min_distance=10):
    # Encourage players to maintain a minimum distance from each other
    spacing_score = 0
    num_players = len(blue_team)
    
    for i in range(num_players):
        for j in range(i + 1, num_players):
            dist = np.linalg.norm(blue_team[i] - blue_team[j])
            # Penalize if distance is less than a threshold
            if dist < min_distance:
                spacing_score -= (min_distance - dist)**2
            else:
                spacing_score += dist**0.5  # small reward for keeping good distance
    
    return spacing_score

def boundaryAvoidanceScore(blue_team, field_size):
    # Encourage players to stay away from the boundaries of the field
    boundary_penalty = 0
    for player in blue_team:
        x, y = player
        dist_to_boundary = min(x, field_size - x, y, field_size - y)
        if dist_to_boundary < 10:  # Assume a buffer zone of 10 units
            boundary_penalty -= (10 - dist_to_boundary)**2
    return -boundary_penalty  # Negative penalty means it's subtracted from fitness

def movementEfficiencyScore(previous_positions, current_positions):
    # Minimize excessive movement; reward for efficient dodging
    efficiency_score = 0
    for prev, curr in zip(previous_positions, current_positions):
        movement_distance = np.linalg.norm(curr - prev)
        efficiency_score -= movement_distance**2  # Penalize unnecessary movement
    return efficiency_score

def threatProximityScore(ball_pos, ball_velocity, blue_team):
    # Penalize players in the direct threat path of the ball
    penalty_score = 0
    for player in blue_team:
        # Vector from ball to player
        to_player_vector = player - ball_pos
        # Normalize vectors
        normalized_ball_velocity = ball_velocity / np.linalg.norm(ball_velocity)
        normalized_to_player = to_player_vector / np.linalg.norm(to_player_vector)
        
        # Calculate cosine similarity
        cosine_similarity = np.dot(normalized_ball_velocity, normalized_to_player)
        
        # If player is in the direct threat path (cosine similarity close to 1)
        if cosine_similarity > 0.9:  # Threshold can be adjusted
            penalty_score -= 100  # Heavier penalty for being in the path
    
    return penalty_score

def totalFitness(ball_pos, ball_velocity, blue_team, previous_positions, field_size):
    # Combine multiple fitness components
    dodge_score = dodgeScore(ball_pos, blue_team)
    spacing_score = teamSpacingScore(blue_team)
    boundary_score = boundaryAvoidanceScore(blue_team, field_size)
    efficiency_score = movementEfficiencyScore(previous_positions, blue_team)
    threat_penalty = threatProximityScore(ball_pos, ball_velocity, blue_team)
    
    # Adjust the weights based on importance
    total_score = (
        1.0 * dodge_score + 
        0.5 * spacing_score + 
        0.3 * boundary_score + 
        0.2 * efficiency_score + 
        1.5 * threat_penalty
    )
    
    return total_score
