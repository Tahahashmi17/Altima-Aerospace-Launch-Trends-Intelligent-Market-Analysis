import pandas as pd

def load_and_clean_data(csv_path):
    df = pd.read_csv(csv_path)

    df['Payload (kg)'] = df['Payload (kg)'].astype(str).str.replace(',', '').astype(float)
    df['Launch Cost ($M)'] = df['Launch Cost ($M)'].astype(str).str.replace(',', '').astype(float)
    df['Price ($/kg)'] = df['Price ($/kg)'].astype(str).str.replace(',', '').astype(float)
    
    df.replace('-', pd.NA, inplace=True)

    df.drop(columns=['Funding ($M)', 'HQ Location'], inplace=True)

    for col in ['Launch Class', 'Orbit Altitude']:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df

def standardize_categories(df):
    def l_class(launch):
        if launch in ['Medium, Heavy','Medium']:
            return 'Medium'
        elif launch in ['Small', 'Small, Medium', 'Small, Heavy']:
            return 'Small'
        elif launch in ['Heavy', 'Heavy, Super Heavy']:
            return 'Heavy'
        elif 'Tourism' in launch:
            return 'Tourism'
        return 'Other'
    df['Launch Class'] = df['Launch Class'].apply(l_class)

    def tech_type(tech):
        if 'Rocket' in tech:
            return 'Rocket'
        elif 'Balloon' in tech:
            return 'Balloon'
        elif 'Spaceplane' in tech:
            return 'Spaceplane'
        else:
            return 'Other'
    df['Tech Type'] = df['Tech Type'].apply(tech_type)

    return df

def map_description_category(desc):
    # Category keywords mapping
    launch_dev_keywords = [
        'developing', 'build', 'nano-launch', 'next generation', 'rocket ship', 'hypersonic'
    ]
    services_keywords = [
        'launch services', 'access to orbit', 'delivery', 'provider', 'transportation', 'space delivery'
    ]
    balloon_keywords = ['balloon', 'ballooning', 'stratospheric']
    tourism_keywords = ['tourism', 'tourist', 'trip into stratosphere', 'space plane']
    satellite_keywords = ['satellite', 'Dream Chaser', 'spacecraft', 'cubesat']
    propulsion_keywords = ['bio-derived', 'RAM-accelerator', 'hybrid', 'electromagnetic', 'sustainable', 'engine']
    
    desc_lower = desc.lower()

    if any(kw in desc_lower for kw in launch_dev_keywords):
        return 'Launch Vehicle Development'
    elif any(kw in desc_lower for kw in services_keywords):
        return 'Launch Services'
    elif any(kw in desc_lower for kw in balloon_keywords):
        return 'Balloon-Based Technologies'
    elif any(kw in desc_lower for kw in tourism_keywords):
        return 'Space Tourism Suborbital'
    elif any(kw in desc_lower for kw in satellite_keywords):
        return 'Satellite Technology and Services'
    elif any(kw in desc_lower for kw in propulsion_keywords):
        return 'Innovative Propulsion Technologies'
    else:
        return 'Space Access and Technology Innovation'

def categorize_description(df):
    df['Description'] = df['Description'].apply(map_description_category)
    return df

def final_cleanup(df):
    # Replace 0 values with mean/median
    df['Payload (kg)'] = df['Payload (kg)'].replace(0, df['Payload (kg)'].mean())
    df['Launch Cost ($M)'] = df['Launch Cost ($M)'].replace(0, df['Launch Cost ($M)'].mean())
    df['SFR'] = df['SFR'].replace(0, df['SFR'].median())

    # Drop unused columns
    df.drop(columns=['Company', 'Country', 'Price ($/kg)'], inplace=True)

    return df
