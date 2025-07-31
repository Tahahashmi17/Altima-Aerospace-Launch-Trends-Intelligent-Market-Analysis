import seaborn as sns
import matplotlib.pyplot as plt
import os

def save_countplot(df, column, title, filename):
    plt.figure(figsize=(10, 6))
    order = df[column].value_counts().index
    sns.countplot(x=column, data=df, order=order, palette="Set2")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"plots/{filename}.png")
    plt.close()

def save_histplot(df, column, title, filename, bins=30):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], bins=bins, kde=True, color="skyblue")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"plots/{filename}.png")
    plt.close()

def run_eda(df):
    os.makedirs("plots", exist_ok=True)

    save_countplot(df, 'Launch Class', "Launches by Launch Class", "launch_class")
    save_countplot(df, 'Tech Type', "Launches by Tech Type", "tech_type")
    save_countplot(df, 'Orbit Altitude', "Launches by Orbit Altitude", "orbit_altitude")
    save_countplot(df, 'Description', "Company Description Categories", "description")

    save_histplot(df, 'Payload (kg)', "Payload Distribution (kg)", "payload_dist")
    save_histplot(df, 'Launch Cost ($M)', "Launch Cost Distribution (Million USD)", "launch_cost_dist")
    save_histplot(df, 'SFR', "SFR Distribution", "sfr_dist", bins=9)
