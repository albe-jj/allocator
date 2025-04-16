import pandas as pd
from datetime import datetime, timedelta
import numpy as np

class Producer:
    """
    Class representing a producer with their attributes and delivery history.
    """
    def __init__(self, id, rating, offered_pallets=None, last_delivery_date=None, delivered_in_program=0):
        """
        Initialize a producer.
        
        Parameters:
        - id: Unique identifier for the producer
        - rating: Producer rating (e.g., 1-5)
        - offered_pallets: Number of pallets offered by the producer for current program
        - last_delivery_date: Date of the last delivery made by this producer
        - delivered_in_program: Total number of pallets delivered
        """
        self.id = id
        self.rating = rating
        self.offered_pallets = offered_pallets  # Number of pallets offered by producer for the current program
        self.last_delivery_date = last_delivery_date
        self.delivered_in_program = delivered_in_program  # Total number of pallets delivered
        self.current_coefficient = 0.0  # Current merit coefficient (calculated during distribution)

    def has_capacity(self, additional_pallets=1):
        """Check if producer has capacity to handle additional pallets."""
        if self.offered_pallets is None:
            return True  # No limit specified
        return (self.delivered_in_program + additional_pallets) <= self.offered_pallets
    
    def remaining_capacity(self):
        """Calculate the remaining capacity."""
        if self.offered_pallets is None:
            return float('inf')  # Unlimited capacity
        return max(0, self.offered_pallets - self.delivered_in_program)
    
    def __str__(self):
        return f"Producer {self.id} (Rating: {self.rating})"


class PalletDistributionSystem:
    def __init__(self, rating_weight=1.0, rotation_weight=0.2, distribution_weight=0.15, current_delivery_weight=0.5):
        """
        Initialize the pallet distribution system with weighting parameters.
        
        Parameters:
        - rating_weight: Weight assigned to producer ratings
        - rotation_weight: Weight assigned to days since last delivery
        - distribution_weight: Weight assigned to pallets already delivered
        - current_delivery_weight: Weight assigned to pallets delivered in current distribution
        """
        self.rating_weight = rating_weight
        self.rotation_weight = rotation_weight
        self.distribution_weight = distribution_weight
        self.current_delivery_weight = current_delivery_weight
    
    def calculate_merit_coefficient(self, producer, current_date, program_total_pallets, current_allocation=None):
        """
        Calculate the merit coefficient for a producer.
        
        Parameters:
        - producer: Producer object
        - current_date: Current date for calculating days since last delivery
        - program_total_pallets: Estimated total pallets for the program (for normalization)
        - current_allocation: Number of pallets already allocated in current distribution
        
        Returns:
        - Merit coefficient value
        """
        # Normalize rating (assuming max rating is 5)
        normalized_rating = producer.rating / 5.0
        
        # Calculate and normalize days since last delivery (capped at 30 days)
        if producer.last_delivery_date:
            days_since_delivery = (current_date - producer.last_delivery_date).days
        else:
            days_since_delivery = 30  # Max value for producers who haven't delivered
        
        # Normalize days since delivery (0 to 1 scale)
        normalized_days = days_since_delivery / 30.0
        
        # Get and normalize pallets already delivered in current program
        pallets_delivered = producer.delivered_in_program
        
        # Avoid division by zero
        if program_total_pallets > 0:
            normalized_pallets = pallets_delivered / program_total_pallets
        else:
            normalized_pallets = 0
            
        # Normalize current distribution allocation (if provided)
        normalized_current_allocation = 0
        if current_allocation is not None and current_allocation > 0:
            # Normalize by the producer's offered pallets instead of a fixed value
            if producer.offered_pallets and producer.offered_pallets > 0:
                normalized_current_allocation = current_allocation / producer.offered_pallets
            else:
                # Fallback if no offered pallets is defined
                print("No offered pallets defined for producer", producer.id)
                normalized_current_allocation = current_allocation / 10.0
        
        # Calculate coefficient using the normalized factors
        rating_contribution = normalized_rating * self.rating_weight
        days_contribution = normalized_days * self.rotation_weight
        delivered_contribution = normalized_pallets * self.distribution_weight
        current_alloc_contribution = normalized_current_allocation * self.current_delivery_weight
        
        coefficient = rating_contribution + days_contribution - delivered_contribution - current_alloc_contribution
        
        # Store debug information in producer object for later display
        producer.debug_info = {
            'rating': producer.rating,
            'rating_norm': normalized_rating,
            'rating_contrib': rating_contribution,
            'days': days_since_delivery,
            'days_norm': normalized_days,
            'days_contrib': days_contribution,
            'delivered': pallets_delivered,
            'delivered_norm': normalized_pallets,
            'delivered_contrib': -delivered_contribution,  # Negative contribution
            'current_alloc': current_allocation or 0,
            'current_alloc_norm': normalized_current_allocation,
            'current_alloc_contrib': -current_alloc_contribution,  # Negative contribution
            'coefficient': coefficient
        }
                      
        return coefficient
    
    def distribute_pallets(self, producers, total_pallets, current_date, 
                           program_total_pallets=None, max_allocation_percentage=0.5):
        """
        Distribute pallets among producers.
        
        Parameters:
        - producers: List of Producer objects
        - total_pallets: Total number of pallets to distribute in this order
        - current_date: Current date
        - program_total_pallets: Estimated total pallets for the entire program (for normalization)
        - max_allocation_percentage: Maximum percentage of total pallets a single producer can receive
        
        Returns:
        - Dictionary with producer IDs as keys and allocated pallets as values
        """
        # If program_total_pallets is not provided, estimate it based on this order and current deliveries
        if program_total_pallets is None:
            current_program_deliveries = sum(p.delivered_in_program for p in producers)
            program_total_pallets = current_program_deliveries + total_pallets
        
        remaining_pallets = total_pallets
        allocation = {p.id: 0 for p in producers}
        max_pallets_per_producer = int(total_pallets * max_allocation_percentage)
        
        # Create a list of eligible producers (those with capacity)
        eligible_producers = [p for p in producers if p.has_capacity()]
        
        # Print warning if any producers are already at max capacity
        ineligible_producers = [p for p in producers if not p.has_capacity()]
        if ineligible_producers:
            print(f"Warning: {len(ineligible_producers)} producers are already at maximum capacity:")
            for p in ineligible_producers:
                print(f"  - {p.id}: at max capacity of {p.offered_pallets}")
        
        # Helper function to print the coefficient table
        def print_coefficient_table(producers_list, iteration=None):
            # Sort producers by coefficient for the debug table
            sorted_producers = sorted(producers_list, key=lambda p: p.debug_info['coefficient'], reverse=True)
            
            if iteration is not None:
                print(f"\n==== Producer Coefficients - Iteration {iteration} ({remaining_pallets} pallets remaining) ====\n")
            else:
                print("\n==== Initial Producer Coefficients (Sorted by Merit) ====\n")
            
            # Display the sorted debug table
            print(f"{'ID':>6} | {'Coef':>6} | {'Rating':>6} | {'R.Cont':>6} | {'Days':>6} | {'D.Cont':>6} | {'Deliv':>6} | {'Dl.Cont':>6} | {'CurAl':>6} | {'CA.Cont':>6}")
            print("-" * 85)
            for p in sorted_producers:
                info = p.debug_info
                print(f"{p.id:>6} | {info['coefficient']:>6.2f} | {info['rating']:>6.1f} | {info['rating_contrib']:>6.2f} | {info['days']:>6d} | {info['days_contrib']:>6.2f} | {info['delivered']:>6d} | {info['delivered_contrib']:>6.2f} | {info['current_alloc']:>6d} | {info['current_alloc_contrib']:>6.2f}")
        
        # Calculate initial coefficients and debug information for display
        for producer in eligible_producers:
            self.calculate_merit_coefficient(
                producer, current_date, program_total_pallets, allocation[producer.id]
            )
        
        # Display initial table with legend
        print_coefficient_table(eligible_producers)
        
        # Display legend (only once)
        print("\nLegend:")
        print("  Coef: Final Merit Coefficient")
        print("  R.Cont: Rating Contribution (positive)")
        print("  D.Cont: Days Since Delivery Contribution (positive)")
        print("  Dl.Cont: Delivered in Program Contribution (negative)")
        print("  CurAl: Current Allocation in this Distribution")
        print("  CA.Cont: Current Allocation Contribution (negative)")
        print("\nWeights used:")
        print(f"  Rating Weight: {self.rating_weight:.2f}")
        print(f"  Rotation Weight: {self.rotation_weight:.2f}")
        print(f"  Distribution Weight: {self.distribution_weight:.2f}")
        print(f"  Current Delivery Weight: {self.current_delivery_weight:.2f}")
        
        print("\nStarting distribution...")
        
        # Keep distributing until all pallets are allocated or no eligible producers remain
        iteration = 0
        while remaining_pallets > 0 and eligible_producers:
            iteration += 1
            
            # Calculate coefficient for each producer
            for producer in eligible_producers:
                producer.current_coefficient = self.calculate_merit_coefficient(
                    producer, current_date, program_total_pallets, allocation[producer.id]
                )
            
            # Sort producers by coefficient (highest first)
            eligible_producers.sort(key=lambda p: p.current_coefficient, reverse=True)
            
            # Allocate a pallet to the producer with highest coefficient
            best_producer = eligible_producers[0]
            allocation[best_producer.id] += 1
            
            # Update the producer's delivered pallets and apply the delivery
            best_producer.delivered_in_program += 1

            # Update the producer's last delivery date to the current date
            best_producer.last_delivery_date = current_date
            
            # Display the updated coefficients after each allocation
            print_coefficient_table(eligible_producers, iteration)
            
            # Remove producer if they've reached the maximum allocation or max capacity
            if (allocation[best_producer.id] >= max_pallets_per_producer or 
                not best_producer.has_capacity()):
                print(f"\nProducer {best_producer.id} removed: {'maximum allocation reached' if allocation[best_producer.id] >= max_pallets_per_producer else 'capacity limit reached'}")
                eligible_producers.remove(best_producer)
            
            remaining_pallets -= 1
            
            # If all producers have reached their max or capacity, report the issue
            if not eligible_producers and remaining_pallets > 0:
                print(f"Warning: All producers reached maximum allocation or capacity.")
                print(f"Unable to distribute {remaining_pallets} pallets.")
                break
        
        print("================================================")
        
        # Show final allocation results in a clear table
        print("\n==== Final Allocation Results ====\n")
        producers.sort(key=lambda p: allocation[p.id], reverse=True)
        
        total_allocated = sum(allocation.values())
        print(f"{'ID':>6} | {'Allocated':>9} | {'Percentage':>10} | {'Total Delivered':>15}")
        print("-" * 50)
        for p in producers:
            pct = (allocation[p.id]/total_pallets*100) if total_pallets > 0 else 0
            print(f"{p.id:>6} | {allocation[p.id]:>9d} | {pct:>9.1f}% | {p.delivered_in_program:>15d}")
        
        return allocation


# Example data and test
def create_example_data():
    """Create example data to test the distribution system."""
    current_date = datetime.now()
    
    # Create example producers with different ratings, capacities, and delivery histories
    producers = [
        Producer(
            id='P001',
            rating=5,
            offered_pallets=20,
            last_delivery_date=current_date - timedelta(days=1),
            delivered_in_program=0
        ),
        Producer(
            id='P002',
            rating=4.5,
            offered_pallets=15,
            last_delivery_date=current_date - timedelta(days=5),
            delivered_in_program=0
        ),
        Producer(
            id='P003',
            rating=4,
            offered_pallets=12,
            last_delivery_date=current_date - timedelta(days=3),
            delivered_in_program=0
        ),
        Producer(
            id='P004',
            rating=3,
            offered_pallets=25,
            last_delivery_date=current_date - timedelta(days=30),
            delivered_in_program=0
        ),
        Producer(
            id='P005',
            rating=2,
            offered_pallets=10,
            last_delivery_date=None,  # New producer
            delivered_in_program=0
        )
    ]
    
    return producers, current_date


def run_example():
    """Run the example distribution scenario."""
    producers, current_date = create_example_data()
    
    # Print initial producer information
    print("Initial producer data:")
    for p in producers:
        delivery_info = f"Last delivery: {p.last_delivery_date.strftime('%Y-%m-%d') if p.last_delivery_date else 'Never'}"
        program_deliveries = p.delivered_in_program
        capacity_info = f"Offered: {p.offered_pallets}, Used: {program_deliveries}, Remaining: {p.remaining_capacity()}"
        print(f"{p.id}: Rating: {p.rating}, {delivery_info}, {capacity_info}")
    
    # Create distribution system
    distribution_system = PalletDistributionSystem(
        rating_weight=1,
        rotation_weight=0.1,
        distribution_weight=1
    )
    
    # Distribute 50 pallets
    total_pallets = 20
    
    # Estimate total program pallets (current + new order)
    current_program_deliveries = sum(p.delivered_in_program for p in producers)
    program_total_pallets = current_program_deliveries + total_pallets
    
    print(f"\nDistributing {total_pallets} pallets (program total estimate: {program_total_pallets})...")
    
    # Calculate initial coefficients for reference
    print("\nInitial merit coefficients:")
    for p in producers:
        coef = distribution_system.calculate_merit_coefficient(p, current_date, program_total_pallets)
        print(f"{p.id}: {coef:.2f}")
    
    # Run the distribution
    allocation = distribution_system.distribute_pallets(
        producers, 
        total_pallets, 
        current_date, 
        program_total_pallets=program_total_pallets,
        max_allocation_percentage=0.4  # No producer can get more than 40% of pallets
    )
    
    # Print results
    print("\nResults:")
    for p in producers:
        pct = (allocation[p.id]/total_pallets*100) if total_pallets > 0 else 0
        print(f"{p.id}: {allocation[p.id]} pallets ({pct:.1f}%)")
    
    # Print updated deliveries and capacities
    print("\nUpdated program deliveries and capacities:")
    for p in producers:
        program_deliveries = p.delivered_in_program
        capacity_status = f"Used: {program_deliveries}/{p.offered_pallets} ({program_deliveries/p.offered_pallets*100:.1f}%)"
        print(f"{p.id}: {program_deliveries} pallets, {capacity_status}")


# def run_example_with_tight_capacity():
#     """Run an example with tighter capacity constraints."""
#     producers, current_date = create_example_data()
#     
#     # Set tighter capacity limits
#     producers[0].offered_pallets = 12  # Producer A already has 10, so only 2 more available
#     producers[1].offered_pallets = 8   # Producer B already has 5, so only 3 more available
#     producers[2].offered_pallets = 10  # Producer C already has 8, so only 2 more available
#     
#     print("\n\n================ Example with Tight Capacity Constraints ================")
#     print("Producer capacity limits:")
#     for p in producers:
#         program_deliveries = p.delivered_in_program
#         remaining = p.remaining_capacity()
#         print(f"{p.id}: Capacity: {p.offered_pallets}, Already delivered: {program_deliveries}, Remaining: {remaining}")
#     
#     # Create distribution system
#     distribution_system = PalletDistributionSystem(
#         rating_weight=1.0,
#         rotation_weight=0.2,
#         distribution_weight=0.15
#     )
#     
#     # Distribute 30 pallets
#     total_pallets = 30
#     
#     # Estimate total program pallets (current + new order)
#     current_program_deliveries = sum(p.delivered_in_program for p in producers)
#     program_total_pallets = current_program_deliveries + total_pallets
#     
#     print(f"\nDistributing {total_pallets} pallets (program total estimate: {program_total_pallets})...")
#     
#     # Run the distribution
#     allocation = distribution_system.distribute_pallets(
#         producers, 
#         total_pallets, 
#         current_date, 
#         program_total_pallets=program_total_pallets,
#         max_allocation_percentage=0.4
#     )


if __name__ == "__main__":
    run_example()
    # run_example_with_tight_capacity()