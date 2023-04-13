import seaborn as sns
import matplotlib.pyplot as plt


def anomaly_plot(test_data, prediction):
    test_data['label'] = prediction
    test1 = test_data[test_data["label"] == 1]
    sns.pairplot(test1[['air_inflow', 'air_end_temp', 'out_pressure', 'air_flow_pressure']])
    sns.pairplot(test1[['motor_current', 'motor_rpm', 'motor_temp', 'motor_vibe']])

    plt.show()


def plot_label_counts(predictions, chunk_sizes):
    for i, (start, end) in enumerate(chunk_sizes):
        chunk = predictions[start:end]
        num_labels = len(chunk[chunk['label'] == 1])
        print(f"Chunk {i}: {num_labels} labels with value 1")
        plt.plot(chunk['label'])
        plt.show()