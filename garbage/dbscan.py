scaler = MinMaxScaler()

train_data = pd.read_csv(r'data\raw\train_data.csv')
train_data = build_features_final.create_derived_features(train_data)

test_data = pd.read_csv(r'data\raw\test_data.csv')
test_data = build_features_final.create_derived_features(test_data)

scaled_train_data = scaler.fit_transform(train_data)
scaled_test_data = scaler.transform(test_data)

scaled_train_data = pd.DataFrame(scaled_train_data, columns=train_data.columns)
scaled_test_data = pd.DataFrame(scaled_test_data, columns=train_data.columns)

scaled_train_data_df = scaled_train_data.copy()
scaled_test_data_df = scaled_test_data.copy()

grouped_train = scaled_train_data.groupby('type')

drop_feature = ['type', 'motor_hp', 'air_end_temp', 'motor_rpm', 'motor_temp', 'motor_vibe', 'motor_current',
                'air_inflow', "air_flow_pressure", "current_by_vibration", "airflow_per_rotation", "air_to_motor_ratio"]

result = np.zeros(len(scaled_test_data_df))

for group_name, group_data in grouped_train:
    test_group = scaled_test_data[scaled_test_data['type'] == group_name]
    train_group = group_data.drop(drop_feature, axis=1)
    test_group = test_group.drop(drop_feature, axis=1)

    dbscan = DBSCAN(eps=0.5, min_samples=5).fit(train_group)

    train_labels = dbscan.labels_
    labels = dbscan.fit_predict(test_group)
    outliers = np.where(labels == -1)[0]
    result[test_group.index[outliers]] = 1
    

    print(f"finish {group_name}type")

anomaly = pd.Series(result)
submission = prediction_to_csv(anomaly)