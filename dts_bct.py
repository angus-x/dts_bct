import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pulp
from datetime import datetime
import io

st.set_page_config(
    page_title="Bed Capacity Planning Tool",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .solver-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    .info-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.8rem;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        width: 280px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'population_data' not in st.session_state:
    st.session_state.population_data = None
if 'admission_data' not in st.session_state:
    st.session_state.admission_data = None
if 'alos_data' not in st.session_state:
    st.session_state.alos_data = None
if 'supply_data' not in st.session_state:
    st.session_state.supply_data = None
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None

def generate_sample_population_data():
    years = [2020, 2025, 2030, 2035, 2040]
    age_bands = ['Age_0_5', 'Age_6_40', 'Age_41_60', 'Age_61_70', 'Age_71_80']
    genders = ['Male', 'Female']
    
    base_population = {
        ('Age_0_5', 'Male'): 18000, ('Age_0_5', 'Female'): 17200,
        ('Age_6_40', 'Male'): 145000, ('Age_6_40', 'Female'): 142000,
        ('Age_41_60', 'Male'): 85000, ('Age_41_60', 'Female'): 82000,
        ('Age_61_70', 'Male'): 32000, ('Age_61_70', 'Female'): 35000,
        ('Age_71_80', 'Male'): 12000, ('Age_71_80', 'Female'): 15000
    }
    
    growth_rates = {
        ('Age_0_5', 'Male'): 0.002, ('Age_0_5', 'Female'): 0.002,
        ('Age_6_40', 'Male'): 0.008, ('Age_6_40', 'Female'): 0.008,
        ('Age_41_60', 'Male'): 0.012, ('Age_41_60', 'Female'): 0.012,
        ('Age_61_70', 'Male'): 0.020, ('Age_61_70', 'Female'): 0.020,
        ('Age_71_80', 'Male'): 0.025, ('Age_71_80', 'Female'): 0.025
    }
    
    rows = []
    for year in years:
        for age_band in age_bands:
            for gender in genders:
                pop = int(base_population[(age_band, gender)] * ((1 + growth_rates[(age_band, gender)]) ** (year - 2020)))
                rows.append({'Year': year, 'Age_Band': age_band, 'Gender': gender, 'Population': pop})
    
    return pd.DataFrame(rows)

def generate_sample_admission_data():
    mdc_groups = ['Cardiovascular', 'Respiratory', 'Digestive', 'Musculoskeletal', 'Nervous System', 'Endocrine']
    age_bands = ['Age_0_5', 'Age_6_40', 'Age_41_60', 'Age_61_70', 'Age_71_80']
    genders = ['Male', 'Female']
    
    base_rates = {
        ('Cardiovascular', 'Age_0_5'): 2, ('Cardiovascular', 'Age_6_40'): 5, ('Cardiovascular', 'Age_41_60'): 15,
        ('Cardiovascular', 'Age_61_70'): 35, ('Cardiovascular', 'Age_71_80'): 55,
        ('Respiratory', 'Age_0_5'): 8, ('Respiratory', 'Age_6_40'): 3, ('Respiratory', 'Age_41_60'): 8,
        ('Respiratory', 'Age_61_70'): 20, ('Respiratory', 'Age_71_80'): 35,
        ('Digestive', 'Age_0_5'): 3, ('Digestive', 'Age_6_40'): 7, ('Digestive', 'Age_41_60'): 12,
        ('Digestive', 'Age_61_70'): 18, ('Digestive', 'Age_71_80'): 25,
    }
    
    rows = []
    for mdc in mdc_groups:
        for age_band in age_bands:
            for gender in genders:
                base_rate = base_rates.get((mdc, age_band), 5)
                rate = base_rate * (1.1 if gender == 'Male' and mdc == 'Cardiovascular' else 1.0)
                rows.append({'MDC_Group': mdc, 'Age_Band': age_band, 'Gender': gender, 'Admission_Rate': rate})
    
    return pd.DataFrame(rows)

def generate_sample_alos_data():
    mdc_groups = ['Cardiovascular', 'Respiratory', 'Digestive', 'Musculoskeletal', 'Nervous System', 'Endocrine']
    age_bands = ['Age_0_5', 'Age_6_40', 'Age_41_60', 'Age_61_70', 'Age_71_80']
    genders = ['Male', 'Female']
    
    base_alos = {
        'Cardiovascular': 5, 'Respiratory': 4, 'Digestive': 3,
        'Musculoskeletal': 4, 'Nervous System': 6, 'Endocrine': 3
    }
    
    rows = []
    for mdc in mdc_groups:
        for age_band in age_bands:
            for gender in genders:
                alos = base_alos[mdc]
                if age_band in ['Age_61_70', 'Age_71_80']:
                    alos += 1
                rows.append({'MDC_Group': mdc, 'Age_Band': age_band, 'Gender': gender, 'ALOS': alos})
    
    return pd.DataFrame(rows)

def generate_sample_supply_data():
    hospitals = [
        {'Institution': 'NUH', 'Type': 'Public', 'Region': 'Western Singapore'},
        {'Institution': 'NTFGH', 'Type': 'Public', 'Region': 'Western Singapore'},
        {'Institution': 'TGCH', 'Type': 'Public', 'Region': 'Western Singapore'},
        {'Institution': 'AIGH', 'Type': 'Public', 'Region': 'Western Singapore'}
    ]
    
    base_capacities = {'NUH': 1200, 'NTFGH': 700, 'TGCH': 400, 'AIGH': 300}
    
    supply_data = []
    for hospital in hospitals:
        institution = hospital['Institution']
        base_2020 = base_capacities[institution]
        growth_rate = 1.025
        
        row = {
            'Institution': institution,
            'Type': hospital['Type'],
            'Region': hospital['Region'],
            'Beds_2020': base_2020,
            'Beds_2025': int(base_2020 * (growth_rate ** 5)),
            'Beds_2030': int(base_2020 * (growth_rate ** 10)),
            'Beds_2035': int(base_2020 * (growth_rate ** 15)),
            'Beds_2040': int(base_2020 * (growth_rate ** 20))
        }
        supply_data.append(row)
    
    return pd.DataFrame(supply_data)

def calculate_demand_from_population(population_df, admission_df, alos_df, target_year, granularity):
    pop_df = population_df[population_df['Year'] == target_year]
    demand_rows = []
    
    if granularity == "Age Band + MDC":
        for _, pop_row in pop_df.iterrows():
            age_band = pop_row['Age_Band']
            gender = pop_row['Gender']
            population = pop_row['Population']
            
            for mdc in admission_df['MDC_Group'].unique():
                adm_row = admission_df[
                    (admission_df['MDC_Group'] == mdc) & 
                    (admission_df['Age_Band'] == age_band) & 
                    (admission_df['Gender'] == gender)
                ]
                alos_row = alos_df[
                    (alos_df['MDC_Group'] == mdc) & 
                    (alos_df['Age_Band'] == age_band) & 
                    (alos_df['Gender'] == gender)
                ]
                
                if not adm_row.empty and not alos_row.empty:
                    admission_rate = adm_row['Admission_Rate'].values[0] / 1000
                    alos = alos_row['ALOS'].values[0]
                    patient_days = population * admission_rate * alos
                    
                    demand_rows.append({
                        'MDC_Group': mdc,
                        'Age_Band': age_band,
                        'Gender': gender,
                        'Patient_Days': int(patient_days)
                    })
    else:  # Age Band Only
        for _, pop_row in pop_df.iterrows():
            age_band = pop_row['Age_Band']
            gender = pop_row['Gender']
            population = pop_row['Population']
            
            total_patient_days = 0
            for mdc in admission_df['MDC_Group'].unique():
                adm_row = admission_df[
                    (admission_df['MDC_Group'] == mdc) & 
                    (admission_df['Age_Band'] == age_band) & 
                    (admission_df['Gender'] == gender)
                ]
                alos_row = alos_df[
                    (alos_df['MDC_Group'] == mdc) & 
                    (alos_df['Age_Band'] == age_band) & 
                    (alos_df['Gender'] == gender)
                ]
                
                if not adm_row.empty and not alos_row.empty:
                    admission_rate = adm_row['Admission_Rate'].values[0] / 1000
                    alos = alos_row['ALOS'].values[0]
                    total_patient_days += population * admission_rate * alos
            
            demand_rows.append({
                'Age_Band': age_band,
                'Gender': gender,
                'Patient_Days': int(total_patient_days)
            })
    
    return pd.DataFrame(demand_rows)

def solve_demand_fulfillment(demand_df, supply_df, target_year, target_bor, granularity):
    try:
        prob = pulp.LpProblem("Demand_Fulfillment_MinMax_BOR", pulp.LpMinimize)
        hospitals = supply_df['Institution'].tolist()
        
        # Create variables
        allocation = {}
        unfulfilled = {}
        
        for _, row in demand_df.iterrows():
            if granularity == "Age Band + MDC":
                key = (row['MDC_Group'], row['Age_Band'], row['Gender'])
            else:
                key = (row['Age_Band'], row['Gender'])
            
            unfulfilled[key] = pulp.LpVariable(f"unfulfilled_{hash(key)}", lowBound=0)
            allocation[key] = {}
            for h in hospitals:
                allocation[key][h] = pulp.LpVariable(f"alloc_{hash(key)}_{h}", lowBound=0)
        
        max_bor = pulp.LpVariable("max_bor", lowBound=0)
        
        # Objective: minimize max BOR + penalty for unfulfilled demand
        unfulfilled_penalty = pulp.lpSum(unfulfilled.values())
        prob += max_bor + 0.001 * unfulfilled_penalty
        
        # Demand constraints
        for _, row in demand_df.iterrows():
            if granularity == "Age Band + MDC":
                key = (row['MDC_Group'], row['Age_Band'], row['Gender'])
            else:
                key = (row['Age_Band'], row['Gender'])
            
            demand = row['Patient_Days']
            fulfilled = pulp.lpSum(allocation[key].values())
            prob += fulfilled + unfulfilled[key] == demand
        
        # Capacity constraints
        supply_col = f'Beds_{target_year}'
        for h in hospitals:
            hospital_beds = supply_df[supply_df['Institution'] == h][supply_col].iloc[0]
            annual_capacity = hospital_beds * 365
            
            total_allocation = pulp.lpSum([allocation[key][h] for key in allocation.keys()])
            prob += total_allocation <= annual_capacity
            
            bor = total_allocation / annual_capacity if annual_capacity > 0 else 0
            prob += max_bor >= bor
        
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=300))
        
        results = {
            'solver_type': 'Demand Fulfillment (Min-Max BOR)',
            'status': pulp.LpStatus[prob.status],
            'hospitals': {},
            'unfulfilled': {},
            'max_bor': 0,
            'total_unfulfilled': 0,
            'fulfillment_rate': 0,
            'granularity': granularity
        }
        
        if prob.status == pulp.LpStatusOptimal:
            results['max_bor'] = pulp.value(max_bor) * 100
            
            total_demand = demand_df['Patient_Days'].sum()
            total_unfulfilled = 0
            
            for h in hospitals:
                hospital_beds = supply_df[supply_df['Institution'] == h][supply_col].iloc[0]
                annual_capacity = hospital_beds * 365
                
                total_allocated = sum([pulp.value(allocation[key][h]) or 0 for key in allocation.keys()])
                bor = (total_allocated / annual_capacity * 100) if annual_capacity > 0 else 0
                
                results['hospitals'][h] = {
                    'beds': hospital_beds,
                    'annual_capacity': annual_capacity,
                    'allocated_days': total_allocated,
                    'bor': bor,
                    'allocations': {}
                }
                
                for key in allocation.keys():
                    results['hospitals'][h]['allocations'][key] = pulp.value(allocation[key][h]) or 0
            
            for key in unfulfilled.keys():
                unfulfilled_amount = pulp.value(unfulfilled[key]) or 0
                if unfulfilled_amount > 0:
                    results['unfulfilled'][key] = unfulfilled_amount
                    total_unfulfilled += unfulfilled_amount
            
            results['total_unfulfilled'] = total_unfulfilled
            results['fulfillment_rate'] = ((total_demand - total_unfulfilled) / total_demand * 100) if total_demand > 0 else 0
        
        return results
    
    except Exception as e:
        return {
            'solver_type': 'Demand Fulfillment (Min-Max BOR)',
            'status': 'Error',
            'error_message': str(e),
            'granularity': granularity
        }

def solve_capacity_maximization(demand_df, supply_df, target_year, target_bor, granularity):
    try:
        prob = pulp.LpProblem("Capacity_Maximization", pulp.LpMaximize)
        hospitals = supply_df['Institution'].tolist()
        
        # Create variables
        allocation = {}
        
        for _, row in demand_df.iterrows():
            if granularity == "Age Band + MDC":
                key = (row['MDC_Group'], row['Age_Band'], row['Gender'])
            else:
                key = (row['Age_Band'], row['Gender'])
            
            allocation[key] = {}
            for h in hospitals:
                allocation[key][h] = pulp.LpVariable(f"alloc_{hash(key)}_{h}", lowBound=0)
        
        excess = {}
        for h in hospitals:
            excess[h] = pulp.LpVariable(f"excess_{h}", lowBound=0)
        
        # Objective: maximize total served - penalty for excess capacity
        total_served = pulp.lpSum([allocation[key][h] for key in allocation.keys() for h in hospitals])
        excess_penalty = pulp.lpSum(excess.values())
        prob += total_served - 0.001 * excess_penalty
        
        # Demand constraints
        for _, row in demand_df.iterrows():
            if granularity == "Age Band + MDC":
                key = (row['MDC_Group'], row['Age_Band'], row['Gender'])
            else:
                key = (row['Age_Band'], row['Gender'])
            
            demand = row['Patient_Days']
            served = pulp.lpSum(allocation[key].values())
            prob += served <= demand
        
        # Capacity constraints
        supply_col = f'Beds_{target_year}'
        for h in hospitals:
            hospital_beds = supply_df[supply_df['Institution'] == h][supply_col].iloc[0]
            annual_capacity = hospital_beds * 365
            max_utilization = annual_capacity * (target_bor / 100)
            
            total_allocation = pulp.lpSum([allocation[key][h] for key in allocation.keys()])
            prob += total_allocation <= max_utilization
            prob += excess[h] == max_utilization - total_allocation
        
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=300))
        
        results = {
            'solver_type': 'Capacity Maximization',
            'status': pulp.LpStatus[prob.status],
            'hospitals': {},
            'total_served': 0,
            'total_excess_capacity': 0,
            'system_bor': 0,
            'granularity': granularity
        }
        
        if prob.status == pulp.LpStatusOptimal:
            supply_col = f'Beds_{target_year}'
            total_served = 0
            total_capacity = 0
            total_excess = 0
            
            for h in hospitals:
                hospital_beds = supply_df[supply_df['Institution'] == h][supply_col].iloc[0]
                annual_capacity = hospital_beds * 365
                max_utilization = annual_capacity * (target_bor / 100)
                
                total_allocated = sum([pulp.value(allocation[key][h]) or 0 for key in allocation.keys()])
                excess_capacity = pulp.value(excess[h]) or 0
                bor = (total_allocated / annual_capacity * 100) if annual_capacity > 0 else 0
                
                results['hospitals'][h] = {
                    'beds': hospital_beds,
                    'annual_capacity': annual_capacity,
                    'max_utilization': max_utilization,
                    'allocated_days': total_allocated,
                    'excess_capacity': excess_capacity,
                    'bor': bor,
                    'allocations': {}
                }
                
                for key in allocation.keys():
                    results['hospitals'][h]['allocations'][key] = pulp.value(allocation[key][h]) or 0
                
                total_served += total_allocated
                total_capacity += annual_capacity
                total_excess += excess_capacity
            
            results['total_served'] = total_served
            results['total_excess_capacity'] = total_excess
            results['system_bor'] = (total_served / total_capacity * 100) if total_capacity > 0 else 0
        
        return results
    
    except Exception as e:
        return {
            'solver_type': 'Capacity Maximization',
            'status': 'Error',
            'error_message': str(e),
            'granularity': granularity
        }

def create_demand_supply_charts(population_df, admission_df, alos_df, supply_df, granularity):
    years = [2020, 2025, 2030, 2035, 2040]
    
    # Calculate demand by year
    demand_by_year = []
    for year in years:
        year_demand = calculate_demand_from_population(population_df, admission_df, alos_df, year, granularity)
        total_demand = year_demand['Patient_Days'].sum()
        demand_by_year.append({'Year': year, 'Total_Demand': total_demand})
    
    demand_timeline_df = pd.DataFrame(demand_by_year)
    
    # Calculate supply by year
    supply_by_year = []
    for year in years:
        if f'Beds_{year}' in supply_df.columns:
            total_beds = supply_df[f'Beds_{year}'].sum()
            annual_capacity = total_beds * 365
            supply_by_year.append({
                'Year': year, 
                'Total_Beds': total_beds,
                'Annual_Capacity': annual_capacity
            })
    
    supply_timeline_df = pd.DataFrame(supply_by_year)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Demand vs Supply Timeline', 'Population Distribution', 
                       'Bed Capacity by Hospital', 'Supply-Demand Gap'),
        specs=[[{"secondary_y": True}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Timeline chart
    if not demand_timeline_df.empty and not supply_timeline_df.empty:
        fig.add_trace(
            go.Scatter(x=demand_timeline_df['Year'], y=demand_timeline_df['Total_Demand'],
                      name='Demand', line=dict(color='red', width=3)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=supply_timeline_df['Year'], y=supply_timeline_df['Annual_Capacity'],
                      name='Supply (100% BOR)', line=dict(color='blue', width=3)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=supply_timeline_df['Year'], y=supply_timeline_df['Annual_Capacity'] * 0.85,
                      name='Supply (85% BOR)', line=dict(color='green', width=3, dash='dash')),
            row=1, col=1
        )
    
    # Population by age and gender
    latest_pop = population_df[population_df['Year'] == population_df['Year'].max()]
    pop_summary = latest_pop.groupby(['Age_Band', 'Gender'])['Population'].sum().reset_index()
    
    for gender in ['Male', 'Female']:
        gender_data = pop_summary[pop_summary['Gender'] == gender]
        fig.add_trace(
            go.Bar(x=gender_data['Age_Band'], y=gender_data['Population'], 
                  name=gender, marker_color='lightblue' if gender == 'Male' else 'pink'),
            row=1, col=2
        )
    
    # Bed capacity by hospital
    if not supply_df.empty:
        latest_year = max([col for col in supply_df.columns if col.startswith('Beds_')])
        blue_colors = ['#08519c', '#3182bd', '#6baed6', '#9ecae1']
        
        fig.add_trace(
            go.Bar(x=supply_df['Institution'], y=supply_df[latest_year],
                  name='Bed Capacity', 
                  marker_color=blue_colors[:len(supply_df)],
                  text=supply_df[latest_year],
                  textposition='outside'),
            row=2, col=1
        )
    
    # Gap analysis
    if not demand_timeline_df.empty and not supply_timeline_df.empty:
        merged_df = pd.merge(demand_timeline_df, supply_timeline_df, on='Year')
        merged_df['Gap_85_BOR'] = merged_df['Total_Demand'] - (merged_df['Annual_Capacity'] * 0.85)
        
        fig.add_trace(
            go.Scatter(x=merged_df['Year'], y=merged_df['Gap_85_BOR'],
                      name='Gap (85% BOR)', line=dict(color='orange', width=3),
                      mode='lines+markers'),
            row=2, col=2
        )
        fig.add_hline(y=0, line_dash="dash", row=2, col=2)
    
    fig.update_layout(height=800, title_text="Bed Capacity Planning - Western Singapore", showlegend=True)
    
    return fig

def main():
    st.title("🏥 Bed Capacity Planning Tool")
    st.markdown("**Advanced demand-supply mapping with dual MILP solvers for bed capacity planning in Western Singapore**")
    
    # Sidebar
    st.sidebar.title("🎛️ Configuration")
    
    st.sidebar.markdown("""
    <div class="info-card">
        <h4>📊 Input Granularity</h4>
        <p>Choose how detailed your analysis should be</p>
    </div>
    """, unsafe_allow_html=True)
    
    granularity = st.sidebar.selectbox(
        "Select granularity:",
        ["Age Band + MDC", "Age Band Only"],
        help="Age Band + MDC: Detailed by medical conditions | Age Band Only: Aggregated across all conditions"
    )
    
    # Navigation
    st.sidebar.markdown("---")
    st.sidebar.title("📍 Navigation")
    page = st.sidebar.selectbox(
        "Go to section:",
        ["📊 Data Management", "📈 Demand & Supply Analysis", "🔧 Optimization & Results"]
    )
    
    if page == "📊 Data Management":
        st.header("📊 Data Management")
        
        st.markdown(f"""
        <div class="info-card">
            <h4>ℹ️ Current Configuration</h4>
            <p><strong>Input Granularity:</strong> {granularity}</p>
            <p><strong>Focus Region:</strong> Western Singapore</p>
            <p><strong>Time Horizon:</strong> 2020-2040</p>
        </div>
        """, unsafe_allow_html=True)
        
        data_source = st.radio("Choose data source:", ["Use Sample Data", "Upload CSV Files"])
        
        if data_source == "Use Sample Data":
            if st.button("🚀 Load Sample Data", type="primary"):
                with st.spinner("Loading sample data..."):
                    st.session_state.population_data = generate_sample_population_data()
                    st.session_state.admission_data = generate_sample_admission_data()
                    st.session_state.alos_data = generate_sample_alos_data()
                    st.session_state.supply_data = generate_sample_supply_data()
                st.success("✅ Sample data loaded successfully!")
        
        else:
            st.subheader("📂 Upload Data Files")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**👥 Population Data**")
                st.caption("Required: Year, Age_Band, Gender, Population")
                pop_file = st.file_uploader("Upload population CSV", type=['csv'], key="pop_upload")
                if pop_file:
                    st.session_state.population_data = pd.read_csv(pop_file)
                    st.success("Population data uploaded!")
                
                st.markdown("**🏥 Admission Rates Data**")
                st.caption(f"Required: {'MDC_Group, ' if granularity == 'Age Band + MDC' else ''}Age_Band, Gender, Admission_Rate")
                admission_file = st.file_uploader("Upload admission rates CSV", type=['csv'], key="admission_upload")
                if admission_file:
                    st.session_state.admission_data = pd.read_csv(admission_file)
                    st.success("Admission data uploaded!")
            
            with col2:
                st.markdown("**📊 ALOS Data**")
                st.caption(f"Required: {'MDC_Group, ' if granularity == 'Age Band + MDC' else ''}Age_Band, Gender, ALOS")
                alos_file = st.file_uploader("Upload ALOS CSV", type=['csv'], key="alos_upload")
                if alos_file:
                    st.session_state.alos_data = pd.read_csv(alos_file)
                    st.success("ALOS data uploaded!")
                
                st.markdown("**🏨 Supply Data**")
                st.caption("Required: Institution, Type, Region, Beds_2020, Beds_2025, etc.")
                supply_file = st.file_uploader("Upload supply CSV", type=['csv'], key="supply_upload")
                if supply_file:
                    st.session_state.supply_data = pd.read_csv(supply_file)
                    st.success("Supply data uploaded!")
        
        # Display data
        if st.session_state.population_data is not None:
            st.subheader("👥 Population Data")
            st.caption("Units: Number of people")
            edited_pop = st.data_editor(st.session_state.population_data, use_container_width=True, key="pop_editor")
            st.session_state.population_data = edited_pop
        if st.session_state.admission_data is not None:
            st.subheader("🏥 Admission Rates")
            st.caption("Units: Admissions per 1,000 population per year")
            edited_admission = st.data_editor(st.session_state.admission_data, use_container_width=True, key="admission_editor")
            st.session_state.admission_data = edited_admission
        if st.session_state.alos_data is not None:
            st.subheader("📊 Average Length of Stay (ALOS)")
            st.caption("Units: Days per admission")
            edited_alos = st.data_editor(st.session_state.alos_data, use_container_width=True, key="alos_editor")
            st.session_state.alos_data = edited_alos
        if st.session_state.supply_data is not None:
            st.subheader("🏨 Bed Supply Data")
            st.caption("Units: Number of beds")
            edited_supply = st.data_editor(st.session_state.supply_data, use_container_width=True, key="supply_editor")
            st.session_state.supply_data = edited_supply

    elif page == "📈 Demand & Supply Analysis":
        st.header("📈 Demand & Supply Analysis")
        
        st.markdown(f"""
        <div class="info-card">
            <h4>📊 Analysis Configuration</h4>
            <p><strong>Granularity:</strong> {granularity}</p>
            <p><strong>Scope:</strong> Western Singapore Public Hospitals</p>
        </div>
        """, unsafe_allow_html=True)
        
        if not all([st.session_state.population_data is not None, 
                   st.session_state.admission_data is not None,
                   st.session_state.alos_data is not None,
                   st.session_state.supply_data is not None]):
            st.warning("⚠️ Please load data first in the Data Management section.")
            return
        
        analysis_year = st.selectbox("Select Year for Analysis:", [2020, 2025, 2030, 2035, 2040])
        
        # Calculate demand
        demand_df = calculate_demand_from_population(
            st.session_state.population_data,
            st.session_state.admission_data,
            st.session_state.alos_data,
            analysis_year,
            granularity
        )
        
        # Overview charts
        overview_fig = create_demand_supply_charts(
            st.session_state.population_data,
            st.session_state.admission_data,
            st.session_state.alos_data,
            st.session_state.supply_data,
            granularity
        )
        st.plotly_chart(overview_fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"📊 Demand Analysis ({analysis_year})")
            st.caption("Units: Patient days per year")
            
            st.dataframe(demand_df.style.format({'Patient_Days': '{:,.0f}'}), use_container_width=True)
            
            if granularity == "Age Band + MDC":
                fig_demand = px.bar(demand_df, x='MDC_Group', y='Patient_Days', color='Age_Band',
                               title=f'Demand by MDC and Age Band ({analysis_year})',
                               color_discrete_sequence=px.colors.qualitative.Set3)
                fig_demand.update_layout(xaxis_tickangle=45)
            else:
                fig_demand = px.bar(demand_df, x='Age_Band', y='Patient_Days', color='Gender',
                               title=f'Demand by Age Band and Gender ({analysis_year})',
                               color_discrete_sequence=['lightblue', 'pink'])
                fig_demand.update_layout(xaxis_tickangle=45)
            
            st.plotly_chart(fig_demand, use_container_width=True)
        
        with col2:
            st.subheader(f"🏨 Supply Analysis ({analysis_year})")
            supply_col = f'Beds_{analysis_year}'
            
            if supply_col in st.session_state.supply_data.columns:
                supply_summary = st.session_state.supply_data[['Institution', supply_col]].copy()
                supply_summary['Annual_Capacity'] = supply_summary[supply_col] * 365
                supply_summary['Effective_Capacity_85%'] = supply_summary['Annual_Capacity'] * 0.85
                
                st.caption("Units: Beds, Patient days per year")
                st.dataframe(
                    supply_summary.style.format({
                        supply_col: '{:,.0f}',
                        'Annual_Capacity': '{:,.0f}',
                        'Effective_Capacity_85%': '{:,.0f}'
                    }),
                    use_container_width=True
                )
                
                blue_colors = ['#08519c', '#3182bd', '#6baed6', '#9ecae1']
                fig_supply = go.Figure(data=[
                    go.Bar(x=st.session_state.supply_data['Institution'],
                          y=st.session_state.supply_data[supply_col],
                          marker_color=blue_colors[:len(st.session_state.supply_data)],
                          text=st.session_state.supply_data[supply_col],
                          textposition='outside')
                ])
                fig_supply.update_layout(
                    title=f'Bed Capacity by Hospital ({analysis_year})',
                    xaxis_tickangle=45,
                    yaxis_title="Number of Beds"
                )
                st.plotly_chart(fig_supply, use_container_width=True)
            else:
                st.warning(f"No supply data available for {analysis_year}")
        
        # Gap analysis
        st.subheader("📊 Supply-Demand Gap Analysis")
        
        if supply_col in st.session_state.supply_data.columns:
            total_demand = demand_df['Patient_Days'].sum()
            total_supply = st.session_state.supply_data[supply_col].sum() * 365
            effective_supply_85 = total_supply * 0.85
            
            gap_metrics = st.columns(4)
            
            with gap_metrics[0]:
                st.metric("Total Demand", f"{total_demand:,.0f}", help="Annual patient days required")
            
            with gap_metrics[1]:
                st.metric("Total Supply (100%)", f"{total_supply:,.0f}", help="Full bed capacity")
            
            with gap_metrics[2]:
                gap = total_demand - effective_supply_85
                color = "normal" if gap <= 0 else "inverse"
                st.metric("Gap at 85% BOR", f"{gap:,.0f}", help="Shortfall in patient days", delta_color=color)
            
            with gap_metrics[3]:
                required_bor = (total_demand / total_supply * 100) if total_supply > 0 else 0
                color = "normal" if required_bor <= 85 else "inverse"
                st.metric("Required BOR", f"{required_bor:.1f}%", help="BOR needed to meet demand", delta_color=color)
    
    elif page == "🔧 Optimization & Results":
        st.header("🔧 Optimization & Results")
        
        st.markdown(f"""
        <div class="info-card">
            <h4>🎯 Optimization Configuration</h4>
            <p><strong>Granularity:</strong> {granularity}</p>
            <p><strong>Solver Options:</strong> Dual MILP algorithms available</p>
        </div>
        """, unsafe_allow_html=True)
        
        if not all([st.session_state.population_data is not None, 
                   st.session_state.admission_data is not None,
                   st.session_state.alos_data is not None,
                   st.session_state.supply_data is not None]):
            st.warning("⚠️ Please load data first in the Data Management section.")
            return
        
        # Parameters
        st.subheader("🎛️ Optimization Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            target_year = st.selectbox("Target Year:", [2025, 2030, 2035, 2040])
        
        with col2:
            target_bor = st.slider("Target BOR (%):", min_value=70, max_value=95, value=85,
                                 help="Bed Occupancy Rate target for optimization")
        
        with col3:
            solver_type = st.selectbox("Solver Type:", 
                                     ["Demand Fulfillment (Min-Max BOR)", "Capacity Maximization"])
        
        # Solver descriptions
        st.subheader("🔧 Solver Options")
        
        solver_col1, solver_col2 = st.columns(2)
        
        with solver_col1:
            st.markdown("""
            <div class="solver-card">
                <h3>🎯 Solver 1: Demand Fulfillment</h3>
                <p><strong>Objective:</strong> Minimize maximum hospital BOR while fulfilling demand</p>
                <ul>
                    <li>Optimizes load balancing across hospitals</li>
                    <li>Identifies unfulfilled demand</li>
                    <li>Minimizes peak utilization</li>
                    <li>Reports detailed allocations</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with solver_col2:
            st.markdown("""
            <div class="solver-card">
                <h3>📈 Solver 2: Capacity Maximization</h3>
                <p><strong>Objective:</strong> Maximize patient days served within BOR constraints</p>
                <ul>
                    <li>Maximizes system throughput</li>
                    <li>Respects BOR limits per hospital</li>
                    <li>Shows excess capacity</li>
                    <li>Optimizes resource utilization</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Run optimization
        if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
            with st.spinner(f"Running {solver_type}..."):
                try:
                    demand_df = calculate_demand_from_population(
                        st.session_state.population_data,
                        st.session_state.admission_data,
                        st.session_state.alos_data,
                        target_year,
                        granularity
                    )
                    
                    if solver_type == "Demand Fulfillment (Min-Max BOR)":
                        results = solve_demand_fulfillment(demand_df, st.session_state.supply_data, 
                                                         target_year, target_bor, granularity)
                    else:
                        results = solve_capacity_maximization(demand_df, st.session_state.supply_data, 
                                                            target_year, target_bor, granularity)
                    
                    st.session_state.optimization_results = results
                    
                    if results['status'] == 'Optimal':
                        st.success("✅ Optimization completed successfully!")
                    else:
                        st.error(f"❌ Optimization failed: {results['status']}")
                        if 'error_message' in results:
                            st.error(f"Error: {results['error_message']}")
                
                except Exception as e:
                    st.error(f"❌ Optimization error: {str(e)}")
        
        # Results display
        if st.session_state.optimization_results is not None:
            results = st.session_state.optimization_results
            
            if results['status'] == 'Optimal':
                st.subheader("📊 Optimization Results")
                
                # Key metrics
                if results['solver_type'] == 'Demand Fulfillment (Min-Max BOR)':
                    metric_cols = st.columns(4)
                    
                    with metric_cols[0]:
                        st.metric("Max BOR", f"{results['max_bor']:.1f}%", 
                                help="Highest BOR among all hospitals")
                    
                    with metric_cols[1]:
                        st.metric("Fulfillment Rate", f"{results['fulfillment_rate']:.1f}%",
                                help="Percentage of demand that can be fulfilled")
                    
                    with metric_cols[2]:
                        st.metric("Total Unfulfilled", f"{results['total_unfulfilled']:,.0f}",
                                help="Patient days that cannot be accommodated")
                    
                    with metric_cols[3]:
                        avg_bor = np.mean([h['bor'] for h in results['hospitals'].values()])
                        st.metric("Average BOR", f"{avg_bor:.1f}%", help="System-wide average BOR")
                
                else:  # Capacity Maximization
                    metric_cols = st.columns(4)
                    
                    with metric_cols[0]:
                        st.metric("Total Served", f"{results['total_served']:,.0f}",
                                help="Patient days that can be served")
                    
                    with metric_cols[1]:
                        st.metric("System BOR", f"{results['system_bor']:.1f}%",
                                help="Overall system bed occupancy rate")
                    
                    with metric_cols[2]:
                        st.metric("Excess Capacity", f"{results['total_excess_capacity']:,.0f}",
                                help="Unused capacity within BOR limits")
                    
                    with metric_cols[3]:
                        # Calculate utilization rate
                        demand_df = calculate_demand_from_population(
                            st.session_state.population_data,
                            st.session_state.admission_data,
                            st.session_state.alos_data,
                            target_year,
                            granularity
                        )
                        total_demand = demand_df['Patient_Days'].sum()
                        utilization_rate = (results['total_served'] / total_demand * 100) if total_demand > 0 else 0
                        st.metric("Demand Utilization", f"{utilization_rate:.1f}%",
                                help="Percentage of total demand being served")
                
                # Hospital BOR analysis
                st.subheader("🏥 Hospital BOR Analysis")
                
                hospital_data = []
                for hospital, data in results['hospitals'].items():
                    hospital_data.append({
                        'Hospital': hospital,
                        'Beds': data['beds'],
                        'Annual Capacity': data['annual_capacity'],
                        'Allocated Days': int(data['allocated_days']),
                        'BOR (%)': data['bor'],
                        'Status': 'High' if data['bor'] > 90 else 'Normal' if data['bor'] > 70 else 'Low'
                    })
                
                hospital_df = pd.DataFrame(hospital_data)
                
                # Color-coded dataframe
                def color_bor(val):
                    if isinstance(val, (int, float)):
                        if val > 90:
                            return 'background-color: #ffcccb'
                        elif val > 80:
                            return 'background-color: #fff2cc'
                        else:
                            return 'background-color: #d4edda'
                    return ''
                
                styled_hospital_df = hospital_df.style.format({
                    'Beds': '{:,.0f}',
                    'Annual Capacity': '{:,.0f}',
                    'Allocated Days': '{:,.0f}',
                    'BOR (%)': '{:.1f}%'
                }).applymap(color_bor, subset=['BOR (%)'])
                
                st.dataframe(styled_hospital_df, use_container_width=True)
                
                # BOR chart
                blue_colors = ['#08519c', '#3182bd', '#6baed6', '#9ecae1']
                
                fig_bor = go.Figure(data=[
                    go.Bar(x=hospital_df['Hospital'],
                          y=hospital_df['BOR (%)'],
                          marker_color=blue_colors[:len(hospital_df)],
                          text=hospital_df['BOR (%)'],
                          texttemplate='%{text:.1f}%',
                          textposition='outside')
                ])
                fig_bor.add_hline(y=target_bor, line_dash="dash", line_color="blue", 
                                 annotation_text=f"Target BOR ({target_bor}%)")
                fig_bor.add_hline(y=90, line_dash="dash", line_color="red", 
                                 annotation_text="High Utilization (90%)")
                fig_bor.update_layout(
                    title='Hospital Bed Occupancy Rates (BOR)',
                    xaxis_tickangle=45,
                    yaxis_title="BOR (%)"
                )
                st.plotly_chart(fig_bor, use_container_width=True)
                
                # Allocation analysis
                st.subheader("📋 Allocation Analysis")
                
                # Create allocation summary
                allocation_data = []
                
                if granularity == "Age Band + MDC":
                    # Summarize by MDC
                    mdc_summary = {}
                    for hospital, data in results['hospitals'].items():
                        for key, allocation in data['allocations'].items():
                            if allocation > 0:
                                mdc = key[0]  # MDC_Group is first element
                                if mdc not in mdc_summary:
                                    mdc_summary[mdc] = {}
                                if hospital not in mdc_summary[mdc]:
                                    mdc_summary[mdc][hospital] = 0
                                mdc_summary[mdc][hospital] += allocation
                    
                    for mdc, hospitals in mdc_summary.items():
                        row = {'Category': mdc}
                        for hospital in results['hospitals'].keys():
                            row[hospital] = hospitals.get(hospital, 0)
                        allocation_data.append(row)
                
                else:
                    # Summarize by Age Band + Gender
                    age_summary = {}
                    for hospital, data in results['hospitals'].items():
                        for key, allocation in data['allocations'].items():
                            if allocation > 0:
                                age_gender = f"{key[0]}_{key[1]}"  # Age_Band_Gender
                                if age_gender not in age_summary:
                                    age_summary[age_gender] = {}
                                if hospital not in age_summary[age_gender]:
                                    age_summary[age_gender][hospital] = 0
                                age_summary[age_gender][hospital] += allocation
                    
                    for age_gender, hospitals in age_summary.items():
                        row = {'Category': age_gender.replace('Age_', '').replace('_', ' ')}
                        for hospital in results['hospitals'].keys():
                            row[hospital] = hospitals.get(hospital, 0)
                        allocation_data.append(row)
                
                if allocation_data:
                    allocation_df = pd.DataFrame(allocation_data)
                    hospital_cols = [col for col in allocation_df.columns if col != 'Category']
                    format_dict = {col: '{:,.0f}' for col in hospital_cols}
                    
                    st.caption("Units: Patient days per year allocated by category")
                    st.dataframe(
                        allocation_df.style.format(format_dict),
                        use_container_width=True
                    )
                    
                    # Allocation heatmap
                    if len(allocation_data) > 1:
                        fig_heatmap = px.imshow(
                            allocation_df.set_index('Category')[hospital_cols].values,
                            x=hospital_cols,
                            y=allocation_df['Category'],
                            color_continuous_scale='Blues',
                            title='Allocation Heatmap (Patient Days)',
                            aspect='auto'
                        )
                        fig_heatmap.update_layout(height=400)
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Export functionality
                st.subheader("💾 Export Results")
                
                export_data = {
                    'Summary': pd.DataFrame([{
                        'Solver_Type': results['solver_type'],
                        'Target_Year': target_year,
                        'Target_BOR': target_bor,
                        'Granularity': granularity,
                        'Status': results['status']
                    }]),
                    'Hospital_BOR': hospital_df,
                    'Demand_Data': calculate_demand_from_population(
                        st.session_state.population_data,
                        st.session_state.admission_data,
                        st.session_state.alos_data,
                        target_year,
                        granularity
                    )
                }
                
                if allocation_data:
                    export_data['Allocation_Matrix'] = allocation_df
                
                # Unfulfilled demand for Demand Fulfillment solver
                if results['solver_type'] == 'Demand Fulfillment (Min-Max BOR)' and results['total_unfulfilled'] > 0:
                    unfulfilled_data = []
                    for key, amount in results['unfulfilled'].items():
                        if granularity == "Age Band + MDC":
                            unfulfilled_data.append({
                                'MDC_Group': key[0],
                                'Age_Band': key[1],
                                'Gender': key[2],
                                'Unfulfilled_Days': amount
                            })
                        else:
                            unfulfilled_data.append({
                                'Age_Band': key[0],
                                'Gender': key[1],
                                'Unfulfilled_Days': amount
                            })
                    
                    if unfulfilled_data:
                        export_data['Unfulfilled_Demand'] = pd.DataFrame(unfulfilled_data)
                
                # Create Excel file
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    for sheet_name, df in export_data.items():
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                excel_data = output.getvalue()
                
                col_export1, col_export2 = st.columns(2)
                
                with col_export1:
                    st.download_button(
                        label="📥 Download Complete Results (Excel)",
                        data=excel_data,
                        file_name=f"Bed_Capacity_Results_{target_year}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col_export2:
                    csv_data = hospital_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download BOR Analysis (CSV)",
                        data=csv_data,
                        file_name=f"Hospital_BOR_Analysis_{target_year}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                # Recommendations
                st.subheader("💡 Recommendations")
                
                recommendations = []
                
                high_bor_hospitals = [h for h, d in results['hospitals'].items() if d['bor'] > 90]
                if high_bor_hospitals:
                    recommendations.append(f"⚠️ **High BOR Alert**: {', '.join(high_bor_hospitals)} operating above 90% BOR. Consider capacity expansion.")
                
                low_bor_hospitals = [h for h, d in results['hospitals'].items() if d['bor'] < 60]
                if low_bor_hospitals:
                    recommendations.append(f"💡 **Underutilized Capacity**: {', '.join(low_bor_hospitals)} have low BOR. Consider service expansion.")
                
                if results['solver_type'] == 'Demand Fulfillment (Min-Max BOR)':
                    if results['fulfillment_rate'] < 95:
                        recommendations.append(f"⚠️ **Capacity Shortage**: Only {results['fulfillment_rate']:.1f}% of demand fulfilled. System expansion needed.")
                    
                    if results['max_bor'] > 95:
                        recommendations.append(f"🚨 **Critical Utilization**: Maximum BOR {results['max_bor']:.1f}% exceeds safe levels.")
                
                for rec in recommendations:
                    st.info(rec)
                
                if not recommendations:
                    st.success("✅ **System Operating Optimally**: No critical issues identified.")
            
            else:
                st.error(f"Optimization failed: {results['status']}")
                if 'error_message' in results:
                    st.error(f"Error details: {results['error_message']}")
    
    # Sidebar info and templates
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 About")
    st.sidebar.markdown("""
    <div class="info-card">
        <h4>🏥 Bed Capacity Planning Tool v5.0</h4>
        <p><strong>Features:</strong></p>
        <ul>
            <li>Dual MILP optimization solvers</li>
            <li>Age bands: 0-5, 6-40, 41-60, 61-70, 71-80</li>
            <li>Western Singapore hospitals focus</li>
            <li>Configurable granularity options</li>
            <li>Advanced allocation analysis</li>
        </ul>
        <p><strong>Built with:</strong> Streamlit & PuLP</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("### 📥 Sample Templates")
    
    if st.sidebar.button("🔄 Generate Templates"):
        sample_pop = generate_sample_population_data()
        sample_admission = generate_sample_admission_data()
        sample_alos = generate_sample_alos_data()
        sample_supply = generate_sample_supply_data()
        
        st.sidebar.download_button(
            "📊 Population Template",
            sample_pop.to_csv(index=False),
            "population_template.csv",
            "text/csv",
            help="Population by year, age band and gender"
        )
        
        st.sidebar.download_button(
            "🏥 Admission Rates Template",
            sample_admission.to_csv(index=False),
            "admission_template.csv",
            "text/csv",
            help="Admission rates per 1000 population"
        )
        
        st.sidebar.download_button(
            "📊 ALOS Template",
            sample_alos.to_csv(index=False),
            "alos_template.csv",
            "text/csv",
            help="Average length of stay data"
        )
        
        st.sidebar.download_button(
            "🏨 Supply Template",
            sample_supply.to_csv(index=False),
            "supply_template.csv",
            "text/csv",
            help="Hospital capacity by year"
        )

if __name__ == "__main__":
    main()