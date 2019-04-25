# Random forest classifier
# Returns plots with feature importance from clustering files.
# For each cluster id 
# frame_ids_sampled and nr_feature_types requires manual input


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

clustering_frame = 6480

#6480/720 + 1 för tid noll
frame_ids_sampled = 10    
# columns in X/ no frame ids - (df.shape[1])/frame_ids_sampled
nr_feature_types = 134 

def calculateImportancePerType(feature_importance_series, nr_feature_types, frame_ids_sampled):
    # Get sum for each feature
    summed_features = [0] * nr_feature_types 
            
    for i in range(0, nr_feature_types):
        for j in range (0, frame_ids_sampled):
            index = i + nr_feature_types*j
            summed_features[i] += feature_importance_series[index]
            #print(feat_importance_series.index[index])
        
    # Get the column names
    col_names = ['P1_Unit_Adept', 'P1_Unit_AdeptPhaseShift', 'P1_Unit_Archon', 'P1_Unit_Assimilator', 'P1_Unit_Carrier', 'P1_Unit_Colossus', 'P1_Unit_CyberneticsCore', 'P1_Unit_DarkShrine', 'P1_Unit_DarkTemplar', 'P1_Unit_Disruptor', 'P1_Unit_DisruptorPhased', 'P1_Unit_FleetBeacon', 'P1_Unit_Forge', 'P1_Unit_Gateway', 'P1_Unit_HighTemplar', 'P1_Unit_Immortal', 'P1_Unit_Interceptor', 'P1_Unit_Mothership', 'P1_Unit_Nexus', 'P1_Unit_Observer', 'P1_Unit_ObserverSurveillanceMode', 'P1_Unit_Oracle', 'P1_Unit_Phoenix', 'P1_Unit_PhotonCannon', 'P1_Unit_Probe', 'P1_Unit_Pylon', 'P1_Unit_RoboticsBay', 'P1_Unit_RoboticsFacility', 'P1_Unit_Sentry', 'P1_Unit_ShieldBattery', 'P1_Unit_Stalker', 'P1_Unit_Stargate', 'P1_Unit_StasisTrap', 'P1_Unit_Tempest', 'P1_Unit_TemplarArchive', 'P1_Unit_TwilightCouncil', 'P1_Unit_VoidRay', 'P1_Unit_WarpGate', 'P1_Unit_WarpPrism', 'P1_Unit_WarpPrismPhasing', 'P1_Unit_Zealot', 'P1_Upgrade_AdeptPiercingAttack', 'P1_Upgrade_BlinkTech', 'P1_Upgrade_CARRIERLAUNCHSPEEDUPGRADE', 'P1_Upgrade_Charge', 'P1_Upgrade_DarkTemplarBlinkUpgrade', 'P1_Upgrade_ExtendedThermalLance', 'P1_Upgrade_GraviticDrive', 'P1_Upgrade_ObserverGraviticBooster', 'P1_Upgrade_PhoenixRangeUpgrade', 'P1_Upgrade_ProtossAirArmorsLevel1', 'P1_Upgrade_ProtossAirArmorsLevel2', 'P1_Upgrade_ProtossAirArmorsLevel3', 'P1_Upgrade_ProtossAirWeaponsLevel1', 'P1_Upgrade_ProtossAirWeaponsLevel2', 'P1_Upgrade_ProtossAirWeaponsLevel3', 'P1_Upgrade_ProtossGroundArmorsLevel1', 'P1_Upgrade_ProtossGroundArmorsLevel2', 'P1_Upgrade_ProtossGroundArmorsLevel3', 'P1_Upgrade_ProtossGroundWeaponsLevel1', 'P1_Upgrade_ProtossGroundWeaponsLevel2', 'P1_Upgrade_ProtossGroundWeaponsLevel3', 'P1_Upgrade_ProtossShieldsLevel1', 'P1_Upgrade_ProtossShieldsLevel2', 'P1_Upgrade_ProtossShieldsLevel3', 'P1_Upgrade_PsiStormTech', 'P1_Upgrade_WarpGateResearch', 'P2_Unit_Adept', 'P2_Unit_AdeptPhaseShift', 'P2_Unit_Archon', 'P2_Unit_Assimilator', 'P2_Unit_Carrier', 'P2_Unit_Colossus', 'P2_Unit_CyberneticsCore', 'P2_Unit_DarkShrine', 'P2_Unit_DarkTemplar', 'P2_Unit_Disruptor', 'P2_Unit_DisruptorPhased', 'P2_Unit_FleetBeacon', 'P2_Unit_Forge', 'P2_Unit_Gateway', 'P2_Unit_HighTemplar', 'P2_Unit_Immortal', 'P2_Unit_Interceptor', 'P2_Unit_Mothership', 'P2_Unit_Nexus', 'P2_Unit_Observer', 'P2_Unit_ObserverSurveillanceMode', 'P2_Unit_Oracle', 'P2_Unit_Phoenix', 'P2_Unit_PhotonCannon', 'P2_Unit_Probe', 'P2_Unit_Pylon', 'P2_Unit_RoboticsBay', 'P2_Unit_RoboticsFacility', 'P2_Unit_Sentry', 'P2_Unit_ShieldBattery', 'P2_Unit_Stalker', 'P2_Unit_Stargate', 'P2_Unit_StasisTrap', 'P2_Unit_Tempest', 'P2_Unit_TemplarArchive', 'P2_Unit_TwilightCouncil', 'P2_Unit_VoidRay', 'P2_Unit_WarpGate', 'P2_Unit_WarpPrism', 'P2_Unit_WarpPrismPhasing', 'P2_Unit_Zealot', 'P2_Upgrade_AdeptPiercingAttack', 'P2_Upgrade_BlinkTech', 'P2_Upgrade_CARRIERLAUNCHSPEEDUPGRADE', 'P2_Upgrade_Charge', 'P2_Upgrade_DarkTemplarBlinkUpgrade', 'P2_Upgrade_ExtendedThermalLance', 'P2_Upgrade_GraviticDrive', 'P2_Upgrade_ObserverGraviticBooster', 'P2_Upgrade_PhoenixRangeUpgrade', 'P2_Upgrade_ProtossAirArmorsLevel1', 'P2_Upgrade_ProtossAirArmorsLevel2', 'P2_Upgrade_ProtossAirArmorsLevel3', 'P2_Upgrade_ProtossAirWeaponsLevel1', 'P2_Upgrade_ProtossAirWeaponsLevel2', 'P2_Upgrade_ProtossAirWeaponsLevel3', 'P2_Upgrade_ProtossGroundArmorsLevel1', 'P2_Upgrade_ProtossGroundArmorsLevel2', 'P2_Upgrade_ProtossGroundArmorsLevel3', 'P2_Upgrade_ProtossGroundWeaponsLevel1', 'P2_Upgrade_ProtossGroundWeaponsLevel2', 'P2_Upgrade_ProtossGroundWeaponsLevel3', 'P2_Upgrade_ProtossShieldsLevel1', 'P2_Upgrade_ProtossShieldsLevel2', 'P2_Upgrade_ProtossShieldsLevel3', 'P2_Upgrade_PsiStormTech', 'P2_Upgrade_WarpGateResearch']
    
    feat_series = pd.Series(summed_features, index = col_names)
    # translate into %, sort and return
    return (feat_series).sort_values(ascending=False)

def calculateTimeImportance(series, nr_feature_types, frame_ids_sampled):
    #print("Time importance")
    summed_time_importance = [0] * frame_ids_sampled
    for i in range(0, frame_ids_sampled):
        #print(i)
        for j in range(0, nr_feature_types):
            index = i*nr_feature_types+j
            summed_time_importance[i] += series[index]
            #print(series.index[index])
    
    #get column names
    col_names = []
    for i in range(0, len(series), nr_feature_types):
        tmp = series.index[i]
        col_names.append(tmp[14:])
       
    return pd.Series(summed_time_importance, index = col_names)
 
def printSeries(series, name, clustering_frame, cluster_id_no):
    series = series.sort_values(ascending=False)
    series = series.nlargest(20)
    plt.rcParams.update({'figure.autolayout': True})
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(series.index.values, series.values)
    ax.set(xlabel='feat importance / max importance', ylabel='feature', title = name + " cluster id: " + str(cluster_id_no) + " clustering frame: " + str(clustering_frame))
    rel_path = "clustering_frame_" + str(clustering_frame) + "/" + str(cluster_id_no) + "/"
    fig.savefig(rel_path + name + ".pdf")

def printConfusionMatrix(y_test, y_pred, name, clustering_frame, cluster_id_no):
    fmt = '.2f' 
    from sklearn.metrics import confusion_matrix
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Only use the labels that appear in the data
    classes = [str(cluster_id_no), '0']

    print(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=name + " cluster id: " + str(cluster_id_no) + " clustering frame: " + str(clustering_frame),
           ylabel='True label',
           xlabel='Predicted label')

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    rel_path = "clustering_frame_" + str(clustering_frame) + "/" + str(cluster_id_no) + "/"
    plt.savefig(rel_path + name + ".png") 
    #np.savetxt(rel_path + name + ".csv", cm, delimiter=",")
    plt.show()
    
def printAllStuffs(X, y, clustering_frame, cluster_id_no, frame_ids_sampled, nr_feature_types):   
    # Create containing folder for resulting plots
    rel_path = "clustering_frame_" + str(clustering_frame) + "/" + str(cluster_id_no) + "/"
    if not os.path.exists(rel_path):
        os.makedirs(rel_path)    
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1848)
    
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    # fit and scale X_train
    X_train = sc.fit_transform(X_train)
    # scale X_test using the same transform
    X_test = sc.transform(X_test)
    
    # Fitting Random Forest Classification to the Training set 
    #try criterion 'entropy' or 'gini'
    from sklearn.ensemble import RandomForestClassifier
    #classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier = RandomForestClassifier(n_jobs = 2, random_state = 0, n_estimators = 100)
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    printConfusionMatrix(y_test, y_pred, "confusion_matrix_",  clustering_frame, cluster_id_no)
    
    # ---- Plot feature importance ----
    feature_importance = classifier.feature_importances_
    # make importances relative to max importance
    
    #Load the feature importances into a pandas series indexed by your column names
    feat_importance_series = pd.Series(feature_importance, index=X.columns)
    printSeries(feat_importance_series, "time dependent feat importance", clustering_frame, cluster_id_no)
    
    # ---- plot time independent feature importance -----
    #6480/720 + 1 för tid noll
    frame_ids_sampled = 10    
    # columns in X/ no frame ids - (df.shape[1])/frame_ids_sampled
    nr_feature_types = 134 
    
    summed_feat_series = calculateImportancePerType(feat_importance_series, nr_feature_types, frame_ids_sampled)
    printSeries(summed_feat_series, "time independent feat importance", clustering_frame, cluster_id_no)
    
    # ---- plot time frame importance ----
    time_importance_series = calculateTimeImportance(feat_importance_series, nr_feature_types, frame_ids_sampled)
    printSeries(time_importance_series, "time importance", clustering_frame, cluster_id_no)

    


# Importing the dataset
df1 = pd.read_csv('newpictureto6480.csv')
df2 = pd.read_csv('6clustering to6480usingcosineandhierarchicalwithwardlinkage.csv')
df2.columns.values[0] = '0Replay_id'

# merge clusterdata with clustering keys to ensure same data
dataset = pd.merge(df1, df2, on='0Replay_id', how = "inner")
# choose cols to work on.
X = dataset.iloc[:, 2:-1]
y = dataset.iloc[:, -1]

nr_of_clusters = y.max()
cluster_range = set(range(1, nr_of_clusters+1))

#cycle through each cluster id, 
#   current id is kept, all the others are set to 0 
#   plot current id vs all other ids 
for i in range(1, nr_of_clusters+1):
    print("current index " + str(i))
    #create set containing all cluster ids except current.
    cluster_range_copy = cluster_range.copy()
    cluster_range_copy.remove(i)
    print(cluster_range_copy)
    y_temp = y
    # replace all but current cluster id numbers with 0
    y_temp = y_temp.replace(to_replace = list(cluster_range_copy), value = 0)
    #print all relevant plots
    print(y_temp)
    printAllStuffs(X,y_temp, clustering_frame, i, frame_ids_sampled, nr_feature_types)
