import pandas as pd     
import numpy as np    
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score      
from sklearn.preprocessing import StandardScaler   
from sklearn.ensemble import RandomForestClassifier    
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve     
import matplotlib.pyplot as plt        
import seaborn as sns        

#chargement des données       
credit_card = pd.read_csv('creditcard.csv')        

#exploration rapide des données     
print(credit_card.head())    
print(credit_card.info())  
print(credit_card['Class'].value_counts())   

#préparation des données       
X = credit_card.drop('Class', axis=1)     
y = credit_card['Class']   

#division en train et test   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)   

#normalisation des données     
scaler = StandardScaler()    
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test)       

#modélisation avec Random Forest  
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)   

#validation croisée avec Stratified K-Fold 
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 
cross_val_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='roc_auc')   
print(f"Scores de validation croisée (ROC-AUC): {cross_val_scores}")   
print(f"Moyenne des scores de validation croisée (ROC-AUC): {cross_val_scores.mean():.2f}")  

#entraînement du modèle  
rf.fit(X_train, y_train)   
      
#prédictions et Évaluation  
y_pred = rf.predict(X_test)  
y_pred_proba = rf.predict_proba(X_test)[:, 1]  

#métriques principales
accuracy = accuracy_score(y_test, y_pred)  
precision = precision_score(y_test, y_pred)  
recall = recall_score(y_test, y_pred)  
f1 = f1_score(y_test, y_pred)  
roc_auc = roc_auc_score(y_test, y_pred_proba)  

print(f"Accuracy: {accuracy:.2f}")   
print(f"Precision: {precision:.2f}")    
print(f"Recall: {recall:.2f}")  
print(f"F1 Score: {f1:.2f}")  
print(f"ROC AUC: {roc_auc:.2f}")  

#matrice de confusion   
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlOrRd')   
plt.title("Matrice de Confusion - Random Forest")   
plt.xlabel("Prédit")   
plt.ylabel("Réel")  
plt.savefig('confusion_matrix_rf.png')   
plt.close()
 
#courbe ROC 
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)   
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})") 
plt.plot([0, 1], [0, 1], linestyle='--') 
plt.xlabel("Taux de Faux Positifs")  
plt.ylabel("Taux de Vrais Positifs")
plt.title("Courbe ROC - Random Forest")             
plt.legend(loc="lower right") 
plt.savefig('roc_curve_rf.png')            
plt.close()
              
#importance des caractéristiques
feature_importance = pd.DataFrame({'Feature': credit_card.drop('Class', axis=1).columns,               
                                   'Importance': rf.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)                    
                   
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))                
plt.title("Top 10 des Caractéristiques les plus importantes - Random Forest")             
plt.savefig('feature_importance_rf.png')       
plt.close()             
