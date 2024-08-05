
from Pinecone_API import PineconeClient
from Neo4j_API import Neo4j_API
from pymining import seqmining
from collections import Counter, defaultdict
import numpy as np
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import pandas as pd

pc_api_key = '79e04bc1-ae29-4c21-83c1-8cbe418ae013'

neo4j = Neo4j_API(username='neo4j',password='')
pc_user = PineconeClient(index_name="oulad-cke-500000",api_key=pc_api_key)

item_list_file_path = 'item_list_oulad.txt'

embeddings = np.load('cke_embeddings(1).npz')

import csv
user_embeddings= embeddings['user_embeddings']
items_embeddings = embeddings['item_embeddings']


with open('user_embeddings.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for embedding in user_embeddings:
        writer.writerow(embedding)


def get_resources_by_user(user_id):
    query = """
        MATCH (u:User)-[w:INTERACTED]->(r:Resource)
        WHERE u.id = $user
        RETURN u.id as user_id, w.time as time, r.id as resource_id
    """

    params = {
        'user': user_id
    }

    df_result = neo4j.run_query(query, params)

    if df_result is not False:
        # Group by user_id and sort by time
        grouped = df_result.groupby('user_id').apply(
            lambda x: x.sort_values('time')['resource_id'].tolist()
        ).to_dict()
        return grouped
    else:
        return []


def get_top_similar_users(target_user,target_user_embedding,k=50):

    most_similar_users = pc_user.query_result(target_user_embedding,k=k)

    most_similar_users_id = [user['id'] for user in most_similar_users]

    if target_user not in most_similar_users_id:
        most_similar_users_id.append(target_user)

    return most_similar_users_id



def get_resources_many_users(users):

    query = """
        MATCH (u:User)-[w:INTERACTED]->(r:Resource)
        WHERE u.id IN $users
        RETURN u.id as user_id, w.time as time, r.id as resource_id
    """

    params = {
        'users': users
    }

    df_result = neo4j.run_query(query, params)
    
    if df_result is not False:
        # Group by user_id and sort by time
        grouped = df_result.groupby('user_id').apply(
            lambda x: x.sort_values('time')['resource_id'].tolist()
        ).to_dict()
        return grouped
    else:
        return {user: [] for user in users}

def generate_similar_users_resources(target_user):

    most_similar_users = get_top_similar_users(target_user)

    print(f"Most similar users: {most_similar_users}")

    users_resources = get_resources_many_users(most_similar_users)

    for user,resources in users_resources.items():
        print(f"User: {user}, Resources: {resources}")


    return users_resources



def check_len_resources(target_user, user_sequences):

    satisfied_users = {}

    for user,sequences in user_sequences.items():

        if (user == target_user or len(sequences)>len(user_sequences[target_user])) and user not in satisfied_users:
            satisfied_users[user] = sequences

    return satisfied_users

def get_subsequences(sequence):
    subsequences = []
    for start in range(len(sequence)):
        for end in range(start + 1, len(sequence) + 1):
            subsequences.append(sequence[start:end])
    return subsequences

def filter_subsequences(subsequences, min_length=2):
    return [subseq for subseq in subsequences if len(subseq) >= min_length]

def get_satisfied_users(target_user,user_sequences):

    # # user_sequences = check_len_resources(target_user,user_sequences)

    # print(list(user_sequences.keys()))
    
    # if len(user_sequences[target_user]) > 1:
    #     ui_subsequences = [user_sequences[target_user][i:] for i in range(len(user_sequences[target_user]))]
    # else:
    #     ui_subsequences = [user_sequences[target_user]]
    # if ui_subsequences:
    #     ui_subsequences = sorted(ui_subsequences, key=len, reverse=True)
    # else:
    #     return []
    # # ui_subsequences = [user_sequences[target_user]]
    # # print(ui_subsequences)    

    # grouped_ui_subsequences = defaultdict(list)
    # for subseq in ui_subsequences:
    #     grouped_ui_subsequences[len(subseq)].append(subseq)

    # uj_results = {}
    # for uj_id,sequence in user_sequences.items():

    #     if uj_id == target_user:
    #         continue
    #     # if len(ui_subsequences)>1: 
    #     #     uj_subsequences = filter_subsequences(get_subsequences(sequence),min_length=1)
    #     # else:
    #     #     uj_subsequences = filter_subsequences(get_subsequences(sequence),min_length=1)
    #     uj_subsequences = filter_subsequences(get_subsequences(sequence),min_length=1)

    #     uj_results[uj_id] =  uj_subsequences

        
    # satisfied_user = {}
    # for length in sorted(grouped_ui_subsequences.keys(), reverse=True):
    #     for ui_subsequence in grouped_ui_subsequences[length]:
    #         for uj in uj_results.keys():
    #             if ui_subsequence in uj_results[uj] and uj not in satisfied_user:
    #                 satisfied_user[uj] = user_sequences[uj]
    #     if satisfied_user:
    #         break

    # satisfied_user[target_user] = user_sequences[target_user]

    return user_sequences

    # return satisfied_user
    
def get_next_items_improved(current_sequence, patterns, sequences):
    next_items = Counter()
    current_set = set(current_sequence)

    # print(f"Current sequence: {current_sequence}")

    for pattern, support in patterns:
        if len(pattern) > len(current_sequence) and pattern[:len(current_sequence)] == current_sequence:
            next_item = pattern[len(current_sequence)]
            if next_item not in current_set:
                next_items[next_item] += support
                # print(f"Adding {next_item} from pattern {pattern} with support {support}")

    for sequence in sequences:
        if len(sequence) > len(current_sequence) and sequence[:len(current_sequence)] == current_sequence:
            next_item = sequence[len(current_sequence)]
            if next_item not in current_set:
                next_items[next_item] += 1
                # print(f"Adding {next_item} from direct sequence {sequence}")

    if not next_items and current_sequence:
        print("No direct next items found, looking for items after last INTERACTED")
        last_INTERACTED = current_sequence[-1]
        for sequence in sequences:
            if last_INTERACTED in sequence:
                idx = sequence.index(last_INTERACTED)
                if idx + 1 < len(sequence):
                    next_item = sequence[idx + 1]
                    next_items[next_item] += 1

    return next_items


def recommend_next_video_improved(target_user, user_videos, target_sequence, top_similar_users, min_support=1, k=3, flag=1):
    recommended_videos = []

    def contains_subsequence(sequence, subsequence):
        it = iter(sequence)
        return all(item in it for item in subsequence)

    while True:
        old_length = len(recommended_videos)
        similar_sequences = [
            tuple(user_videos[user]) for user in top_similar_users 
            if user != target_user and any(contains_subsequence(user_videos[user], target_sequence[i:]) for i in range(len(target_sequence)))
        ]

        print(f"Similar Sequences: {similar_sequences}")
        
        # Filter and truncate similar sequences to include only those containing the target subsequences
        filtered_sequences = []
        if flag == 1:
            max_length = 5  # Set a maximum length for the truncated sequences
        elif flag == 2:
            max_length = 10
        else:
            max_length = 15
        for seq in similar_sequences:
            for i in range(len(seq)):
                if contains_subsequence(seq[i:], target_sequence):
                    truncated_seq = seq[i:i+max_length]
                    filtered_sequences.append(truncated_seq)
                    break

        print(f"Filtered and Truncated Sequences: {filtered_sequences}")
        
        patterns = seqmining.freq_seq_enum(filtered_sequences, min_support)
        print("Frequent patterns:", list(patterns))

        next_items = get_next_items_improved(target_sequence, patterns, similar_sequences)
        print("Next possible items (with scores):", dict(next_items))

        recommended_video = next_items.most_common(1)
        if recommended_video and recommended_video[0] not in recommended_videos:
            recommended_videos.append(recommended_video[0])
            target_sequence.append(recommended_video[0][0])
        
        if len(recommended_videos) == k or len(recommended_videos) == old_length:
            break

    return recommended_videos


def generate_new_user_embedding(new_user_interactions, item_lists, item_embeddings):
    embedding_list = []

    for interaction in new_user_interactions:

        map_id = item_lists.loc[item_lists['org_id'] == interaction, 'remap_id']


        if not map_id.empty:
            item_embedding = item_embeddings[map_id.values[0]]
            embedding_list.append(item_embedding)
        else:
            print(f'Warning: Interaction {interaction} not found in item list.')

    if not embedding_list:
        return []
    
    # Aggregate embeddings to generate new user embedding
    new_user_embedding = np.mean(embedding_list, axis=0)

    return new_user_embedding




def main(new_user):

    item_lists = pd.read_csv(item_list_file_path, sep='\s+', header=0)

    new_user_interactions = get_resources_by_user(new_user)[new_user]

    len_new_user_interactions = len(new_user_interactions)



    print(new_user_interactions)

    new_user_embedding = generate_new_user_embedding(
                                        new_user_interactions=new_user_interactions,
                                        item_lists=item_lists,
                                        item_embeddings=items_embeddings)

    print(new_user_embedding)
    top_similar_users=get_top_similar_users(new_user,new_user_embedding.tolist())

    print(top_similar_users)
    similar_users_videos = get_resources_many_users(top_similar_users)

    satisfied_users_videos = get_satisfied_users(new_user,similar_users_videos)

    print("Satisfied videos: ",satisfied_users_videos)
    rcm_LOs = recommend_next_video_improved(
                                        target_user=new_user,
                                        user_videos=satisfied_users_videos,
                                        target_sequence=new_user_interactions,
                                        top_similar_users=list(satisfied_users_videos.keys()),
                                        k=len_new_user_interactions-0)
    return rcm_LOs


def test(user_id,flag):
    item_lists = pd.read_csv(item_list_file_path, sep='\s+', header=0)

    
    new_user_interactions = get_resources_by_user(user_id)[user_id]
    len_new_user_interactions = len(new_user_interactions)

    if len_new_user_interactions <= 1:
        return  (None,None) 

    midpoint = len_new_user_interactions // 2
    train_interactions = new_user_interactions[:midpoint]
    print(train_interactions)
    test_interactions = new_user_interactions[midpoint:]
    print(test_interactions)
    new_user_embedding = generate_new_user_embedding(
        new_user_interactions=train_interactions,
        item_lists=item_lists,
        item_embeddings=items_embeddings
    )

    
    if len(new_user_embedding)>0:

        top_similar_users = get_top_similar_users(user_id, new_user_embedding.tolist())
        similar_users_videos = get_resources_many_users(top_similar_users)
        satisfied_users_videos = get_satisfied_users(user_id, similar_users_videos)
        print(satisfied_users_videos)
        rcm_LOs = recommend_next_video_improved(
            target_user=user_id,
            user_videos=satisfied_users_videos,
            target_sequence=train_interactions,
            top_similar_users=list(satisfied_users_videos.keys()),
            k=len_new_user_interactions - midpoint,
            flag=flag
        )

        recommended_resources = [lo[0] for lo in rcm_LOs]

        return recommended_resources, test_interactions
    else:
        return (None,None) 

def evaluate_recommendations(recommended_resources, test_interactions):
    if len(recommended_resources) != len(test_interactions):
        print("Recommended resources:", recommended_resources)
        print("Test interactions:", test_interactions)
        print("Length of recommended resources and test interactions are not the same")
        return (None, None)
    
    accuracy = 0
    recall_hits = set()

    # Calculate accuracy and identify hits for recall calculation
    for i, item in enumerate(recommended_resources):
        if item == test_interactions[i]:
            accuracy += 1
        if item in test_interactions:
            recall_hits.add(item)

    # Calculate accuracy@k
    accuracy_at_k = accuracy / len(test_interactions) if len(test_interactions) else 0
    
    # Calculate recall@k
    recall_at_k = len(recall_hits) / len(test_interactions) if len(test_interactions) else 0

    return accuracy_at_k, recall_at_k

def run_evaluation(sample_users,flag):

    accuracy_scores = []
    recall_scores = []

    for user_id in sample_users:
        print(f"Processing user: {user_id}")
        recommended_resources, test_interactions = test(user_id,flag=flag)

        if recommended_resources and test_interactions:
            accuracy,recall = evaluate_recommendations(recommended_resources, test_interactions)
            
            if not accuracy and not recall:
                continue
            accuracy_scores.append(accuracy)
            recall_scores.append(recall)
            print(f"User: {user_id}, Accuracy: {accuracy}, Recall: {recall}")


    avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0

    return  avg_recall,avg_accuracy

def get_user_and_number_resources(users):

    query = """
        MATCH (u:User)-[w:INTERACTED]->(r:Resource)
        WHERE u.user_id IN $users
        WITH u, count(r.resource_id) as resources
        RETURN resources, count(u.user_id) as user_count
        ORDER BY resources
    """

    params = {
        'users': users
    }

    df_result = neo4j.run_query(query, params)

    if df_result is not False:
        df_result.to_csv('users_resources_number.csv',index=False)
        # Group by user_id and sort by time
        return df_result
    else:
        return []
    
def get_users_from_group_1():
    query = """
        MATCH (u:User)-[w:INTERACTED]->(r:Resource)
        WITH u, count(r.id) as resources
        WHERE resources = 10
        RETURN u.id as users
    """

    df_result = neo4j.run_query(query)

    if df_result is not False:
        # Group by user_id and sort by time
        return df_result
    else:
        return []
    
def get_users_from_group_2(users):
    query = """
        MATCH (u:User)-[w:INTERACTED]->(r:Resource)
        WHERE u.user_id IN $users
        WITH u, count(r.resource_id) as resources
        WHERE resources > 5 AND resources <= 10
        RETURN u.user_id as users
    """

    params = {
        'users': users
    }

    df_result = neo4j.run_query(query, params)

    if df_result is not False:
        # Group by user_id and sort by time
        return df_result
    else:
        return [] 
    
def get_users_from_group_3(users):
    query = """
        MATCH (u:User)-[w:INTERACTED]->(r:Resource)
        WHERE u.user_id IN $users
        WITH u, count(r.resource_id) as resources
        WHERE resources > 10
        RETURN u.user_id as users
    """

    params = {
        'users': users
    }

    df_result = neo4j.run_query(query, params)

    if df_result is not False:
        # Group by user_id and sort by time
        return df_result
    else:
        return []

def plot_histogram(df_result):
    import matplotlib.pyplot as plt
    import seaborn as sns
    if not df_result.empty:
        # Ensure that the dataframe has the correct columns
        if 'resources' in df_result.columns and 'user_count' in df_result.columns:
            sns.histplot(df_result, x='resources', weights='user_count', bins=30, kde=False)
            plt.xlabel('Number of Resources INTERACTED')
            plt.ylabel('Number of Users')
            plt.title('Distribution of Users by Number of Resources INTERACTED')
            plt.show()
        else:
            print("DataFrame does not contain the expected columns.")
    else:
        print("No data to plot.")


def plot_histogram_with_percentage(df_result):
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not df_result.empty:
        # Ensure that the dataframe has the correct columns
        if 'resources' in df_result.columns and 'user_count' in df_result.columns:
            # Calculate the total number of users
            total_users = df_result['user_count'].sum()
            
            # Normalize user counts to percentages
            df_result['percentage'] = (df_result['user_count'] / total_users) * 100
            
            # Plot the histogram
            sns.histplot(df_result, x='resources', weights='percentage', bins=30, kde=False)
            plt.xlabel('Number of Resources INTERACTED')
            plt.ylabel('Percentage of Users')
            plt.title('Distribution of Users by Number of Resources INTERACTED (as Percentage)')
            plt.show()
        else:
            print("DataFrame does not contain the expected columns.")
    else:
        print("No data to plot.")



if __name__ == '__main__':

#     sample_users = [
# "U_1002784","U_100281","U_10033817","U_1004662","U_1004670","U_1004954","U_1005254","U_1005500","U_10057731","U_1006254","U_10068981","U_10073375","U_10073419","U_1008050","U_1008642","U_1008653","U_1009173","U_1009481","U_10100704","U_10137498","U_10137508","U_10137510","U_10138438","U_10167679","U_10170061","U_10170744","U_10172329","U_10174787","U_10181205","U_10187253","U_10187261","U_10187267","U_10192575","U_10192740","U_10192777","U_10192955","U_10193096","U_10193468","U_10193984","U_10194022","U_10195298","U_10197595","U_10199145","U_1020100","U_1020293","U_10212422","U_1021609","U_1021628","U_10222631","U_10222675","U_10225349","U_1022796","U_10231045","U_10231066","U_10236096","U_10236159","U_10236189","U_1023803","U_10244776","U_10244788","U_10244811","U_10244818","U_10244931","U_10246891","U_10246958","U_10247830","U_10248516","U_10248791","U_10249738","U_10255613","U_10261732","U_10261942","U_10262024","U_10272961","U_1027395","U_10275637","U_10275641","U_10275648","U_10275656","U_10275663","U_10275672","U_10275680","U_10275685","U_10290533","U_10290902","U_10290952","U_10291424","U_10291671","U_10291924","U_10292278","U_10292660","U_10304917","U_10305912","U_10306796","U_10322104","U_10323283","U_10323297","U_10323300","U_10340055","U_10340066"    ]

    import random
    import time
    
    df = pd.read_csv("data.csv")

    USERS = df['user_id'].tolist()
    
    print(USERS)

    users_group_1 = get_users_from_group_1()['users'].tolist()
    # print(users_group_1)

    # exit()
    print(len(users_group_1))


    experiment_times = 0

    top_50_random_users_from_group_1 = []
    top_50_random_users_from_group_2 = []
    top_50_random_users_from_group_3 = []

    accuracy_group_1 = []
    recall_group_1 = []

    selected_users = []

    avg_recall_final = []
    avg_accuracy_final = []
    executed_time = []
    
    while experiment_times < 1:
        start_time = time.time()
        # users_group_1 = [user for user in users_group_1 if user not in selected_users]
        
        # top_50_random_users_from_group_1 = random.choices(users_group_1,k=50)
        # print(f"Top 50 random: {top_50_random_users_from_group_1}")
        # selected_users.append(top_50_random_users_from_group_1)
        # top_50_random_users_from_group_2 = random.choices(users_group_2,k=50)
        # top_50_random_users_from_group_3 = random.choices(users_group_3,k=50)
        
        avg_recall,avg_accuracy = run_evaluation(users_group_1,3)



        end_time = time.time()

        avg_recall_final.append(avg_recall)
        avg_accuracy_final.append(avg_accuracy)
        executed_time.append(end_time-start_time)

        print(f"AVG Recall: {avg_recall}")
        print(f"AVG Accuracy: {avg_accuracy}")
        print(f"AVG Execution Time: {end_time-start_time}")
        time.sleep(30)
        experiment_times+=1

    print("AVG Recall Final: ",sum(avg_recall_final)/len(avg_recall_final))

    print("AVG Accuracy Final: ",sum(avg_accuracy_final)/len(avg_accuracy_final))

    print("AVG Execution Time: ",sum(executed_time)/len(executed_time))


    # ks = [10,20,30,40,50]
    # # avg_precisions = []
    # # avg_recalls = []
    # avg_accuracies = []

    # for k in ks:
    #     selected_users = TOP_2279_USERS[:k]
    #     avg_accuracy = run_evaluation(selected_users)
   
    #     avg_accuracies.append(avg_accuracy)
    #     print(f"k={k}, Average Accuracy: {avg_accuracy}")

    # # Plotting
    # plt.figure(figsize=(10, 6))
    # # plt.plot(ks, avg_precisions, label='Average Precision', marker='o')
    # # plt.plot(ks, avg_recalls, label='Average Recall', marker='o')
    # plt.plot(ks, avg_accuracies, label='Average Accuracy', marker='o')
    # plt.xlabel('Number of Sample Users (k)')
    # plt.ylabel('Scores')
    # plt.title('Accuracy vs. Number of Sample Users')
    # plt.xticks(ks)  # Set the ticks for the x-axis
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    




    



