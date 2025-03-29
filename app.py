import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta, date
import numpy as np
from pallet_distributor import Producer, PalletDistributionSystem
import matplotlib.colors as mcolors

# Set page configuration
st.set_page_config(
    page_title="Pallet Distribution Simulator",
    page_icon="📦",
    layout="wide"
)

# Convert date to datetime if needed
def ensure_datetime(d):
    """Convert date to datetime if it's a date object"""
    if isinstance(d, date) and not isinstance(d, datetime):
        return datetime.combine(d, datetime.min.time())
    return d

# Initialize session state to store data
if 'producers' not in st.session_state:
    # Initialize with 5 producers by default
    current_date = datetime.now()
    
    # Create 5 default producers
    st.session_state.producers = [
        Producer(
            id='P001',
            rating=5.0,  # Ensure this is a float
            max_capacity=20,
            last_delivery_date=current_date - timedelta(days=1),
            delivered_in_program=0
        ),
        Producer(
            id='P002',
            rating=4.5,  # Ensure this is a float
            max_capacity=15,
            last_delivery_date=current_date - timedelta(days=5),
            delivered_in_program=0
        ),
        Producer(
            id='P003',
            rating=4.0,  # Ensure this is a float
            max_capacity=12,
            last_delivery_date=current_date - timedelta(days=3),
            delivered_in_program=0
        ),
        Producer(
            id='P004',
            rating=3.0,  # Ensure this is a float
            max_capacity=25,
            last_delivery_date=current_date - timedelta(days=30),
            delivered_in_program=0
        ),
        Producer(
            id='P005',
            rating=2.0,  # Ensure this is a float
            max_capacity=10,
            last_delivery_date=None,  # New producer
            delivered_in_program=0
        )
    ]
    
if 'current_date' not in st.session_state:
    st.session_state.current_date = datetime.now()

# Initialize state to track changes
if 'last_distribution_params' not in st.session_state:
    st.session_state.last_distribution_params = {
        'rating_weight': 1.0,
        'rotation_weight': 0.15,
        'distribution_weight': 1.0,
        'total_pallets': 20,
        'max_allocation_percentage': 0.4,
        'producer_hash': None
    }

# Function to calculate a hash of producer data to detect changes
def calculate_producer_hash(producers):
    data_string = ""
    for p in producers:
        data_string += f"{p.id}|{p.rating}|{p.max_capacity}|{p.delivered_in_program}|"
        if p.last_delivery_date:
            data_string += f"{p.last_delivery_date.timestamp()}|"
        else:
            data_string += "None|"
    return hash(data_string)

def initialize_demo_data():
    """Initialize with demo data for quick testing"""
    current_date = datetime.now()
    
    # Create example producers
    producers = [
        Producer(
            id='P001',
            rating=5.0,  # Ensure this is a float
            max_capacity=20,
            last_delivery_date=current_date - timedelta(days=1),
            delivered_in_program=0
        ),
        Producer(
            id='P002',
            rating=4.5,  # Ensure this is a float
            max_capacity=15,
            last_delivery_date=current_date - timedelta(days=5),
            delivered_in_program=0
        ),
        Producer(
            id='P003',
            rating=4.0,  # Ensure this is a float
            max_capacity=12,
            last_delivery_date=current_date - timedelta(days=3),
            delivered_in_program=0
        ),
        Producer(
            id='P004',
            rating=3.0,  # Ensure this is a float
            max_capacity=25,
            last_delivery_date=current_date - timedelta(days=30),
            delivered_in_program=0
        ),
        Producer(
            id='P005',
            rating=2.0,  # Ensure this is a float
            max_capacity=10,
            last_delivery_date=None,  # New producer
            delivered_in_program=0
        )
    ]
    
    st.session_state.producers = producers
    st.session_state.current_date = current_date

# Main app title
st.title("🚚 Pallet Distribution Simulator")
st.write("Simulate and visualize pallet distribution among producers based on merit coefficients.")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Load demo data button
    if st.button("Reset to Default Data"):
        initialize_demo_data()
    
    # Distribution system parameters
    st.subheader("Distribution Parameters")
    rating_weight = st.slider("Rating Weight", 0.0, 2.0, 1.0, 0.1)
    rotation_weight = st.slider("Rotation Weight (days since delivery)", 0.0, 1.0, 0.15, 0.05)
    distribution_weight = st.slider("Distribution Weight (already delivered)", 0.0, 2.0, 1.0, 0.1)
    
    # Allocation parameters
    st.subheader("Allocation Parameters")
    total_pallets = st.number_input("Total Pallets to Distribute", min_value=1, value=20, step=1)
    max_allocation_percentage = st.slider("Max Allocation per Producer (%)", 10, 100, 40, 5) / 100
    
    # Clear all data button
    if st.button("Clear All Data"):
        st.session_state.producers = []

# Main area
tab1, tab2 = st.tabs(["Producer Distribution", "About"])

# Tab 1: Combined Producers Management and Distribution Results
with tab1:
    st.header("Producer Management")
    
    # Display current producers in a table
    if st.session_state.producers:
        # Calculate current date once
        current_date = ensure_datetime(st.session_state.current_date)
        
        producer_data = []
        for p in st.session_state.producers:
            # Calculate days since last delivery
            if p.last_delivery_date:
                days_since_delivery = (current_date - ensure_datetime(p.last_delivery_date)).days
            else:
                days_since_delivery = None  # Never delivered
            
            producer_data.append({
                "ID": p.id,
                "Rating": p.rating,
                "Max Capacity": p.max_capacity,
                "Days Since Delivery": days_since_delivery,
                "Delivered In Program": p.delivered_in_program,
                "Remaining Capacity": p.remaining_capacity()
            })
        
        producer_df = pd.DataFrame(producer_data)
        
        # Define column configurations for the data editor
        column_config = {
            "ID": st.column_config.TextColumn(
                "ID",
                disabled=True,  # ID should not be editable
                help="Producer unique identifier"
            ),
            "Rating": st.column_config.NumberColumn(
                "Rating",
                min_value=1.0,
                max_value=5.0,
                step=0.5,
                format="%.1f",
                help="Producer rating (1.0-5.0)"
            ),
            "Max Capacity": st.column_config.NumberColumn(
                "Max Capacity",
                min_value=1,
                step=1,
                format="%d",
                help="Maximum capacity for this producer"
            ),
            "Days Since Delivery": st.column_config.NumberColumn(
                "Days Since Delivery",
                min_value=0,
                step=1,
                format="%d",
                help="Days since last delivery (empty means never delivered)"
            ),
            "Delivered In Program": st.column_config.NumberColumn(
                "Delivered In Program",
                min_value=0,
                step=1,
                format="%d",
                help="Number of pallets already delivered in this program"
            ),
            "Remaining Capacity": st.column_config.NumberColumn(
                "Remaining Capacity",
                disabled=True,  # This is a calculated field
                format="%d",
                help="Remaining capacity (Max Capacity - Delivered)"
            )
        }
        
        # Use data_editor instead of dataframe
        edited_df = st.data_editor(
            producer_df,
            column_config=column_config,
            use_container_width=True,
            num_rows="fixed",
            hide_index=True
        )
        
        # Check if the dataframe was edited
        if not edited_df.equals(producer_df):
            # Update the producer objects with the edited values
            for i, row in edited_df.iterrows():
                producer = next((p for p in st.session_state.producers if p.id == row["ID"]), None)
                if producer:
                    # Update the editable fields
                    producer.rating = float(row["Rating"])
                    producer.max_capacity = int(row["Max Capacity"])
                    producer.delivered_in_program = int(row["Delivered In Program"])
                    
                    # Handle the days since delivery field
                    if pd.notna(row["Days Since Delivery"]):
                        days = int(row["Days Since Delivery"])
                        # Calculate the date based on days since delivery
                        producer.last_delivery_date = current_date - timedelta(days=days)
                    else:
                        producer.last_delivery_date = None
            
            # Force recalculation of distribution by clearing the cached results
            if 'last_distribution_results' in st.session_state:
                del st.session_state.last_distribution_results
            
            st.success("Producer data updated!")
        
        # Distribution Results Section
        st.header("Distribution Results")
        
        # Get the current parameters
        current_distribution_params = {
            'rating_weight': rating_weight,
            'rotation_weight': rotation_weight,
            'distribution_weight': distribution_weight,
            'total_pallets': total_pallets,
            'max_allocation_percentage': max_allocation_percentage,
            'producer_hash': calculate_producer_hash(st.session_state.producers)
        }
        
        # Check if any parameters have changed
        parameters_changed = False
        for key, value in current_distribution_params.items():
            if key not in st.session_state.last_distribution_params or st.session_state.last_distribution_params[key] != value:
                parameters_changed = True
                break
        
        # Store current parameters for next comparison
        st.session_state.last_distribution_params = current_distribution_params.copy()
        
        # Automatically run distribution when parameters change
        if parameters_changed or 'last_distribution_results' not in st.session_state:
            # Create distribution system
            distribution_system = PalletDistributionSystem(
                rating_weight=rating_weight,
                rotation_weight=rotation_weight,
                distribution_weight=distribution_weight
            )
            
            # Estimate total program pallets
            current_program_deliveries = sum(p.delivered_in_program for p in st.session_state.producers)
            program_total_pallets = current_program_deliveries + total_pallets
            
            # Calculate initial coefficients
            initial_coefficients = {}
            # Make sure date is datetime for consistency
            current_date = ensure_datetime(st.session_state.current_date)
            for p in st.session_state.producers:
                initial_coefficients[p.id] = distribution_system.calculate_merit_coefficient(
                    p, current_date, program_total_pallets
                )
            
            # Make a copy of the producers to avoid changing the originals during simulation
            simulation_producers = []
            for p in st.session_state.producers:
                sim_producer = Producer(
                    id=p.id,
                    rating=float(p.rating),  # Ensure this is a float
                    max_capacity=p.max_capacity,
                    last_delivery_date=p.last_delivery_date,
                    delivered_in_program=p.delivered_in_program
                )
                simulation_producers.append(sim_producer)
            
            # Run the distribution
            allocation = distribution_system.distribute_pallets(
                simulation_producers, 
                total_pallets, 
                current_date,  # Use the consistent datetime
                program_total_pallets=program_total_pallets,
                max_allocation_percentage=max_allocation_percentage
            )
            
            # Create result dataframe
            result_data = []
            for p in simulation_producers:
                allocated = allocation[p.id]
                percent = (allocated/total_pallets*100) if total_pallets > 0 else 0
                initial_coef = initial_coefficients[p.id]
                
                result_data.append({
                    "ID": p.id,
                    "Rating": p.rating,
                    "Initial Merit": round(initial_coef, 2),
                    "Pallets Allocated": allocated,
                    "Percentage": f"{percent:.1f}%",
                    "Previous Delivery": p.delivered_in_program - allocated,
                    "New Total": p.delivered_in_program,
                    "Capacity Used": f"{p.delivered_in_program}/{p.max_capacity} ({p.delivered_in_program/p.max_capacity*100 if p.max_capacity else 0:.1f}%)"
                })
            
            # Store results in session state
            st.session_state.last_distribution_results = {
                'result_data': result_data,
                'allocation': allocation,
                'simulation_producers': simulation_producers,
                'initial_coefficients': initial_coefficients,
                'program_total_pallets': program_total_pallets
            }
        else:
            # Use cached results
            result_data = st.session_state.last_distribution_results['result_data']
            allocation = st.session_state.last_distribution_results['allocation']
            simulation_producers = st.session_state.last_distribution_results['simulation_producers']
            initial_coefficients = st.session_state.last_distribution_results['initial_coefficients']
            program_total_pallets = st.session_state.last_distribution_results['program_total_pallets']
        
        # Display results
        st.subheader("Distribution Summary")
        result_df = pd.DataFrame(result_data)
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        
        # Create visualizations
        st.subheader("Visualizations")
        
        # Helper function for custom percentage and count labels
        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                val = int(round(pct*total/100.0))
                return f'{pct:.1f}%\n({val} pallets)'
            return my_autopct
        
        # Generate a consistent colormap for producers
        producer_ids = [p["ID"] for p in result_data]
        # Create a colormap with distinct colors - using a different method to specify the colormap
        color_palette = plt.get_cmap('tab10')
        colors = [color_palette(i % 10) for i in range(len(producer_ids))]
        color_map = dict(zip(producer_ids, colors))
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart of current allocation percentages
            st.write("Current Allocation Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Filter only producers who received allocations
            producers_with_allocation = [p for p in result_data if p["Pallets Allocated"] > 0]
            labels = [p["ID"] for p in producers_with_allocation]
            sizes = [p["Pallets Allocated"] for p in producers_with_allocation]
            
            # Get the matching colors for these producers
            allocation_colors = [color_map[producer_id] for producer_id in labels]
            
            if sum(sizes) > 0:  # Ensure we have data to plot
                wedges, texts, autotexts = ax.pie(
                    sizes, 
                    labels=labels, 
                    colors=allocation_colors,
                    autopct=make_autopct(sizes),
                    startangle=90,
                    textprops={'fontsize': 11}
                )
                
                # Improve label visibility
                for text, autotext in zip(texts, autotexts):
                    text.set_fontsize(12)
                    autotext.set_fontsize(10)
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                plt.title("Current Distribution", fontsize=14)
                
                st.pyplot(fig)
            else:
                st.info("No pallets were allocated.")
        
        with col2:
            # Pie chart of program total distribution
            st.write("Total Program Distribution (including current allocation)")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get total program deliveries for each producer
            program_totals = [p["New Total"] for p in result_data]
            total_program_pallets = sum(program_totals)
            
            # Only include producers with any program deliveries
            producers_with_program_deliveries = []
            program_delivery_sizes = []
            program_delivery_colors = []
            
            for i, p in enumerate(result_data):
                if p["New Total"] > 0:
                    producers_with_program_deliveries.append(p["ID"])
                    program_delivery_sizes.append(p["New Total"])
                    program_delivery_colors.append(color_map[p["ID"]])
            
            if sum(program_delivery_sizes) > 0:  # Ensure we have data to plot
                wedges, texts, autotexts = ax.pie(
                    program_delivery_sizes, 
                    labels=producers_with_program_deliveries, 
                    colors=program_delivery_colors,
                    autopct=make_autopct(program_delivery_sizes),
                    startangle=90,
                    textprops={'fontsize': 11}
                )
                
                # Improve label visibility
                for text, autotext in zip(texts, autotexts):
                    text.set_fontsize(12)
                    autotext.set_fontsize(10)
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                
                ax.axis('equal')
                plt.title(f"Program Total: {total_program_pallets} pallets", fontsize=14)
                
                st.pyplot(fig)
            else:
                st.info("No pallets have been delivered in the program.")
        
        # Option to apply the distribution
        if st.button("Apply This Distribution"):
            # Update the actual producers with the simulation results
            for p in st.session_state.producers:
                sim_p = next((sp for sp in simulation_producers if sp.id == p.id), None)
                if sim_p:
                    p.delivered_in_program = sim_p.delivered_in_program
            
            # Force recalculation next time
            if 'last_distribution_results' in st.session_state:
                del st.session_state.last_distribution_results
            
            st.success("Distribution applied to producers!")
            st.rerun()
    else:
        st.info("No producers added yet. Click 'Reset to Default Data' in the sidebar.")

# Tab 2: About
with tab2:
    st.header("About this App")
    st.write("""
    This application simulates pallet distribution among producers based on a merit-based system.
    
    ### How it works:
    
    1. **Producers**: Each producer has attributes like ID, rating, capacity, and delivery history.
    
    2. **Merit Coefficient**: The system calculates a merit coefficient for each producer based on:
       - **Producer Rating**: Higher rated producers get priority
       - **Days Since Last Delivery**: Producers who haven't delivered recently get higher priority (rotation)
       - **Current Program Deliveries**: Producers who have already received many pallets get lower priority (fair distribution)
    
    3. **Distribution Algorithm**:
       - Calculates merit coefficients for all eligible producers
       - Assigns one pallet to the producer with the highest coefficient
       - Reduces that producer's coefficient by the decay factor
       - Continues until all pallets are distributed or no eligible producers remain
    
    ### Parameters:
    
    - **Rating Weight**: How much emphasis to place on producer rating
    - **Rotation Weight**: How much emphasis to place on days since last delivery
    - **Distribution Weight**: How much emphasis to place on existing deliveries (negative factor)
    - **Max Allocation Percentage**: Maximum percentage of total pallets any single producer can receive
    - **Decay Factor**: How quickly a producer's coefficient decreases after receiving a pallet
    
    ### Use Cases:
    
    - Distributing orders fairly among qualified producers
    - Ensuring rotation of producers while maintaining quality standards
    - Balancing workload across your producer network
    """)

# Run the app with: streamlit run app.py 