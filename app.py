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
    page_title="Simulatore di Distribuzione Pallet",
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
    # Initialize with 7 producers by default
    current_date = datetime.now()
    
    # Create 7 default producers
    st.session_state.producers = [
        Producer(
            id='P001',
            rating=5.0,  # Ensure this is a float
            offered_pallets=3,
            last_delivery_date=current_date - timedelta(days=6),
            delivered_in_program=0
        ),
        Producer(
            id='P002',
            rating=4.5,  # Ensure this is a float
            offered_pallets=3,
            last_delivery_date=current_date - timedelta(days=5),
            delivered_in_program=0
        ),
        Producer(
            id='P003',
            rating=4.0,  # Ensure this is a float
            offered_pallets=5,
            last_delivery_date=current_date - timedelta(days=4),
            delivered_in_program=0
        ),
        Producer(
            id='P004',
            rating=3.5,  # Ensure this is a float
            offered_pallets=2,
            last_delivery_date=current_date - timedelta(days=3),
            delivered_in_program=0
        ),
        Producer(
            id='P005',
            rating=3.0,  # Ensure this is a float
            offered_pallets=7,
            # last_delivery_date=None,  # New producer
            last_delivery_date=current_date - timedelta(days=7),
            delivered_in_program=0
        ),
        Producer(
            id='P006',
            rating=2.5,  # Ensure this is a float
            offered_pallets=5,
            last_delivery_date=current_date - timedelta(days=10),
            delivered_in_program=0
        ),
        Producer(
            id='P007',
            rating=2.0,  # Ensure this is a float
            offered_pallets=3,
            last_delivery_date=current_date - timedelta(days=11),
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
        'current_delivery_weight': 0.5,
        'total_pallets': 10,
        'program_total_pallets': 10,
        'max_allocation_percentage': 0.4,
        'producer_hash': None
    }

# Function to calculate a hash of producer data to detect changes
def calculate_producer_hash(producers):
    data_string = ""
    for p in producers:
        data_string += f"{p.id}|{p.rating}|{p.offered_pallets}|{p.delivered_in_program}|"
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
            offered_pallets=15,
            last_delivery_date=current_date - timedelta(days=1),
            delivered_in_program=0
        ),
        Producer(
            id='P002',
            rating=4.5,  # Ensure this is a float
            offered_pallets=15,
            last_delivery_date=current_date - timedelta(days=5),
            delivered_in_program=0
        ),
        Producer(
            id='P003',
            rating=4.0,  # Ensure this is a float
            offered_pallets=15,
            last_delivery_date=current_date - timedelta(days=3),
            delivered_in_program=0
        ),
        Producer(
            id='P004',
            rating=3.5,  # Ensure this is a float
            offered_pallets=15,
            last_delivery_date=current_date - timedelta(days=30),
            delivered_in_program=0
        ),
        Producer(
            id='P005',
            rating=3.0,  # Ensure this is a float
            offered_pallets=15,
            last_delivery_date=None,  # New producer
            delivered_in_program=0
        ),
        Producer(
            id='P006',
            rating=2.5,  # Ensure this is a float
            offered_pallets=15,
            last_delivery_date=current_date - timedelta(days=15),
            delivered_in_program=0
        ),
        Producer(
            id='P007',
            rating=2.0,  # Ensure this is a float
            offered_pallets=15,
            last_delivery_date=current_date - timedelta(days=10),
            delivered_in_program=0
        )
    ]
    
    st.session_state.producers = producers
    st.session_state.current_date = current_date

# Main app title
st.title("🚚 Simulatore di Distribuzione Pallet")
st.write("Simula e visualizza la distribuzione di pallet tra i produttori in base ai coefficienti di merito.")

# Sidebar for configuration
with st.sidebar:
    st.header("Configurazione")
    
    # Load demo data button
    if st.button("Ripristina Dati Predefiniti"):
        initialize_demo_data()
    
    # Distribution system parameters
    st.subheader("Parametri di Distribuzione")
    rating_weight = st.slider("Peso del Rating", 0.0, 2.0, 1.0, 0.1)
    rotation_weight = st.slider("Peso della Rotazione (giorni dall'ultima consegna)", 0.0, 1.0, 0.75, 0.05)
    distribution_weight = st.slider("Peso della Distribuzione (pallet già consegnati nel programma)", 0.0, 2.0, 1.0, 0.1)
    current_delivery_weight = st.slider("Peso della Distribuzione Corrente", 0.0, 4.0, 2.0, 0.1, 
                                      help="Riduce il coefficiente in base ai pallet già allocati nella distribuzione corrente, normalizzati rispetto alla capacità massima del produttore")
    
    # Allocation parameters
    st.subheader("Parametri di Allocazione")
    total_pallets = st.number_input("Totale Pallet da Distribuire", min_value=1, value=10, step=1)
    program_total_pallets = st.number_input("Totale Pallet del Programma", min_value=1, value=10, step=1, 
                                            help="Stima del totale dei pallet per l'intero programma (per il calcolo del coefficiente)")
    max_allocation_percentage = st.slider("Allocazione Massima per Produttore (%)", 10, 100, 40, 5) / 100
    
    # Clear all data button
    if st.button("Cancella Tutti i Dati"):
        st.session_state.producers = []

# Main area
tab1, tab2 = st.tabs(["Distribuzione Produttori", "Informazioni"])

# Tab 1: Combined Producers Management and Distribution Results
with tab1:
    st.header("Gestione Produttori")
    
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
                "Valutazione": p.rating,
                "Pallet Offerti": p.offered_pallets,
                "Giorni Dall'Ultima Consegna": days_since_delivery,
                "Consegnati Nel Programma": p.delivered_in_program,
                "Capacità Rimanente": p.remaining_capacity()
            })
        
        producer_df = pd.DataFrame(producer_data)
        
        # Define column configurations for the data editor
        column_config = {
            "ID": st.column_config.TextColumn(
                "ID",
                disabled=True,  # ID should not be editable
                help="Identificatore unico del produttore"
            ),
            "Valutazione": st.column_config.NumberColumn(
                "Valutazione",
                min_value=1.0,
                max_value=5.0,
                step=0.5,
                format="%.1f",
                help="Valutazione del produttore (1.0-5.0)"
            ),
            "Pallet Offerti": st.column_config.NumberColumn(
                "Pallet Offerti",
                min_value=1,
                step=1,
                format="%d",
                help="Numero di pallet offerti da questo produttore"
            ),
            "Giorni Dall'Ultima Consegna": st.column_config.NumberColumn(
                "Giorni Dall'Ultima Consegna",
                min_value=0,
                step=1,
                format="%d",
                help="Giorni dall'ultima consegna (vuoto significa mai consegnato)"
            ),
            "Consegnati Nel Programma": st.column_config.NumberColumn(
                "Consegnati Nel Programma",
                min_value=0,
                step=1,
                format="%d",
                help="Numero di pallet già consegnati in questo programma"
            ),
            "Capacità Rimanente": st.column_config.NumberColumn(
                "Capacità Rimanente",
                disabled=True,  # This is a calculated field
                format="%d",
                help="Capacità rimanente (Pallet Offerti - Consegnati)"
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
                    producer.rating = float(row["Valutazione"])
                    producer.offered_pallets = int(row["Pallet Offerti"])
                    producer.delivered_in_program = int(row["Consegnati Nel Programma"])
                    
                    # Handle the days since delivery field
                    if pd.notna(row["Giorni Dall'Ultima Consegna"]):
                        days = int(row["Giorni Dall'Ultima Consegna"])
                        # Calculate the date based on days since delivery
                        producer.last_delivery_date = current_date - timedelta(days=days)
                    else:
                        producer.last_delivery_date = None
            
            # Force recalculation of distribution by clearing the cached results
            if 'last_distribution_results' in st.session_state:
                del st.session_state.last_distribution_results
            
            st.success("Dati del produttore aggiornati!")
            st.rerun()  # Force Streamlit to rerun the app with updated data
        
        # Distribution Results Section
        st.header("Risultati della Distribuzione")
        
        # Get the current parameters
        current_distribution_params = {
            'rating_weight': rating_weight,
            'rotation_weight': rotation_weight,
            'distribution_weight': distribution_weight,
            'current_delivery_weight': current_delivery_weight,
            'total_pallets': total_pallets,
            'program_total_pallets': program_total_pallets,
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
                distribution_weight=distribution_weight,
                current_delivery_weight=current_delivery_weight
            )
            
            # Use the explicitly provided program_total_pallets instead of calculating it
            current_program_deliveries = sum(p.delivered_in_program for p in st.session_state.producers)
            
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
                    offered_pallets=p.offered_pallets,
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
                    "Valutazione": p.rating,
                    "Merito Iniziale": round(initial_coef, 2),
                    "Pallet Allocati": allocated,
                    "Percentuale": f"{percent:.1f}%",
                    "Consegna Precedente": p.delivered_in_program - allocated,
                    "Nuovo Totale": p.delivered_in_program,
                    "Capacità Utilizzata": f"{p.delivered_in_program}/{p.offered_pallets} ({p.delivered_in_program/p.offered_pallets*100 if p.offered_pallets else 0:.1f}%)"
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
        st.subheader("Riepilogo della Distribuzione")
        result_df = pd.DataFrame(result_data)
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        
        # Create visualizations
        st.subheader("Visualizzazioni")
        
        # Helper function for custom percentage and count labels
        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                val = int(round(pct*total/100.0))
                return f'{pct:.1f}%\n({val} pallet)'
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
            st.write("Distribuzione dell'Allocazione Corrente")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Filter only producers who received allocations
            producers_with_allocation = [p for p in result_data if p["Pallet Allocati"] > 0]
            labels = [p["ID"] for p in producers_with_allocation]
            sizes = [p["Pallet Allocati"] for p in producers_with_allocation]
            
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
                plt.title("Distribuzione Corrente", fontsize=14)
                
                st.pyplot(fig)
            else:
                st.info("Nessun pallet è stato allocato.")
        
        with col2:
            # Pie chart of program total distribution
            st.write("Distribuzione Totale del Programma (inclusa l'allocazione corrente)")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get total program deliveries for each producer
            program_totals = [p["Nuovo Totale"] for p in result_data]
            total_program_pallets = sum(program_totals)
            
            # Only include producers with any program deliveries
            producers_with_program_deliveries = []
            program_delivery_sizes = []
            program_delivery_colors = []
            
            for i, p in enumerate(result_data):
                if p["Nuovo Totale"] > 0:
                    producers_with_program_deliveries.append(p["ID"])
                    program_delivery_sizes.append(p["Nuovo Totale"])
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
                plt.title(f"Totale Programma: {total_program_pallets} pallet", fontsize=14)
                
                st.pyplot(fig)
            else:
                st.info("Nessun pallet è stato consegnato nel programma.")
        
        # Option to apply the distribution
        if st.button("Applica Questa Distribuzione"):
            # Update the actual producers with the simulation results
            for p in st.session_state.producers:
                sim_p = next((sp for sp in simulation_producers if sp.id == p.id), None)
                if sim_p:
                    p.delivered_in_program = sim_p.delivered_in_program
            
            # Force recalculation next time
            if 'last_distribution_results' in st.session_state:
                del st.session_state.last_distribution_results
            
            st.success("Distribuzione applicata ai produttori!")
            st.rerun()
    else:
        st.info("Nessun produttore aggiunto ancora. Clicca 'Ripristina Dati Predefiniti' nella barra laterale.")

# Tab 2: About
with tab2:
    st.header("Informazioni sull'App")
    st.write("""
    Questa applicazione simula la distribuzione di pallet tra produttori basata su un sistema di merito.
    
    ### Come funziona:
    
    1. **Produttori**: Ogni produttore ha attributi come ID, valutazione, capacità e cronologia delle consegne.
    
    2. **Coefficiente di Merito**: Il sistema calcola un coefficiente di merito per ogni produttore basato su:
       - **Valutazione del Produttore**: I produttori con valutazioni più alte hanno priorità
       - **Giorni Dall'Ultima Consegna**: I produttori che non hanno effettuato consegne di recente hanno priorità più alta (rotazione)
       - **Consegne Attuali del Programma**: I produttori che hanno già ricevuto molti pallet hanno priorità inferiore (distribuzione equa)
    
    3. **Algoritmo di Distribuzione**:
       - Calcola i coefficienti di merito per tutti i produttori idonei
       - Assegna un pallet al produttore con il coefficiente più alto
       - Riduce il coefficiente di quel produttore mediante il fattore di decadimento
       - Continua fino a quando tutti i pallet sono distribuiti o non rimangono produttori idonei
    
    ### Parametri:
    
    - **Peso della Valutazione**: Quanto enfatizzare la valutazione del produttore
    - **Peso della Rotazione**: Quanto enfatizzare i giorni dall'ultima consegna
    - **Peso della Distribuzione**: Quanto enfatizzare le consegne esistenti (fattore negativo)
    - **Allocazione Massima per Produttore**: Percentuale massima del totale dei pallet che un singolo produttore può ricevere
    - **Fattore di Decadimento**: Quanto rapidamente il coefficiente di un produttore diminuisce dopo aver ricevuto un pallet
    
    ### Casi d'Uso:
    
    - Distribuzione equa degli ordini tra produttori qualificati
    - Garanzia di rotazione dei produttori mantenendo gli standard di qualità
    - Bilanciamento del carico di lavoro nella rete di produttori
    """)

# Run the app with: streamlit run app.py 