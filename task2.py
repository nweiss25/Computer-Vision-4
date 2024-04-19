# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 23:02:01 2024

@author: Nathan Weiss
"""

from deepface import DeepFace
import pandas as pd

def main():
    
    val_csv_path = 'fairface_label_val.csv'
    df = pd.read_csv(val_csv_path)
    #df = df.reset_index()
    
    race_occur_dict = {}
    race_correct_dict = {}
    race_abs_error = {}
    gen_occur_dict = {}
    gen_correct_dict = {}
    gen_abs_error = {}
    
    total_correct = 0
    total = 0
    total_abs_error = 0
    
    
    for index, row in df.iterrows():
        
        if index % 500 == 0 and index != 0:
            print("Age Accuracy after",index,"images:", total_correct/total)
            
        age_pred = DeepFace.analyze(row['file'], actions = ['age'], enforce_detection=False)
        
        
        correct = False
        dist = 0
        if (row['age'] == "more than 70"):
            correct = age_pred[0]['age'] >= 70
            if not correct:
                dist = 70 - age_pred[0]['age']
        else: 
            n1, n2 = (int(s) for s in row['age'].split('-'))
            above = age_pred[0]['age']>=n2
            below = age_pred[0]['age']<=n1
            if above:
                dist = age_pred[0]['age'] - n2
            elif below:
                dist = n1 - age_pred[0]['age']
            correct = not(above or below)
            
                
    
        total += 1
        total_abs_error += dist
        r = row['race']
        g = row['gender']
        
        if correct:
            total_correct += 1
            if r in race_correct_dict:
                race_correct_dict[r] += 1
            else:
                race_correct_dict[r] = 1
            if g in gen_correct_dict:
                gen_correct_dict[g] += 1
            else:
                gen_correct_dict[g] = 1
        
        if r in race_occur_dict:
            race_occur_dict[r] += 1
        else:
            race_occur_dict[r] = 1
        if g in gen_occur_dict:
            gen_occur_dict[g] += 1
        else:
            gen_occur_dict[g] = 1
            
        if r in race_abs_error:
            race_abs_error[r] += dist
        else:
            race_abs_error[r] = dist
        if g in gen_abs_error:
            gen_abs_error[g] += dist
        else:
            gen_abs_error[g] = dist
    
    print("\n\nOverall Accuracy:", str(round(total_correct/total*100,3))+"%")
    print("Overall MAE:", round(total_abs_error/total,2))
    print("Total num samples:",total, end="\n\n" )
    
    race_percent_correct = {} 
    print("--------Accuracy by Race:--------")
    for r in race_correct_dict:
        race_percent_correct[r] = race_correct_dict[r] / race_occur_dict[r]
        print("Accuracy for",r,"individuals:", str(round(race_percent_correct[r]*100,3))+"%")
        print("MAE for", r, "individuals:", round(race_abs_error[r] / race_occur_dict[r],2))
        print("Number of",r,"samples:", race_occur_dict[r])
        print()
    
    gen_percent_correct = {}

    print("\n--------Accuracy by Gender:--------")
    for g in gen_correct_dict:
        gen_percent_correct[g] = gen_correct_dict[g] / gen_occur_dict[g]
        print("Accuracy for",g,"individuals:", str(round(gen_percent_correct[g]*100,3)) + "%")
        print("MAE for", g, "individuals:", round(gen_abs_error[g] / gen_occur_dict[g],2))
        print("Number of",g,"samples:", gen_occur_dict[g])
        print()
        
    #print(gen_percent_correct)
    #print(race_percent_correct)    
    
if __name__ == "__main__":
    main()