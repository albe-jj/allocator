import pandas as pd
from datetime import datetime, timedelta
import numpy as np

class Producer:
    """
    Class representing a producer with their attributes and delivery history.
    """
    def __init__(self, id, rating, max_capacity=None, last_delivery_date=None, delivered_in_program=0):
        """
        Initialize a producer.
        
        Parameters:
        - id: Unique identifier for the producer
        - rating: Producer rating (e.g., 1-5)
        - max_capacity: Maximum number of pallets the producer can handle for current program
        - last_delivery_date: Date of the last delivery made by this producer
        - delivered_in_program: Total number of pallets delivered
        """
        self.id = id
        self.rating = rating
        self.max_capacity = max_capacity  # Maximum pallets producer can handle for the current program
        self.last_delivery_date = last_delivery_date
        self.delivered_in_program = delivered_in_program  # Total number of pallets delivered
        self.current_coefficient = 0.0  # Current merit coefficient (calculated during distribution)

    def has_capacity(self, additional_pallets=1):
        """Check if producer has capacity to handle additional pallets."""
        if self.max_capacity is None:
            return True  # No limit specified
        return (self.delivered_in_program + additional_pallets) <= self.max_capacity
    
    def remaining_capacity(self):
        """Calculate the remaining capacity."""
        if self.max_capacity is None:
            return float('inf')  # Unlimited capacity
        return max(0, self.max_capacity - self.delivered_in_program)
    
    def __str__(self):
        return f"Producer {self.id} (Rating: {self.rating})"


class PalletDistributionSystem:
    def __init__(self, rating_weight=1.0, rotation_weight=0.2, distribution_weight=0.15):
        """
        Initialize the pallet distribution system with weighting parameters.
        
        Parameters:
        - rating_weight: Weight assigned to producer ratings
        - rotation_weight: Weight assigned to days since last delivery
        - distribution_weight: Weight assigned to pallets already delivered
        """
        self.rating_weight = rating_weight
        self.rotation_weight = rotation_weight
        self.distribution_weight = distribution_weight
    
    def calculate_merit_coefficient(self, producer, current_date, program_total_pallets):
        """
        Calculate the merit coefficient for a producer.
        
        Parameters:
        - producer: Producer object
        - current_date: Current date for calculating days since last delivery
        - program_total_pallets: Estimated total pallets for the program (for normalization)
        
        Returns:
        - Merit coefficient value
        """
        # Normalize rating (assuming max rating is 5)
        normalized_rating = producer.rating / 5.0
        
        # Calculate and normalize days since last delivery (capped at 30 days)
        if producer.last_delivery_date:
            days_since_delivery = min(30, (current_date - producer.last_delivery_date).days)
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
        
        # Calculate coefficient using the normalized factors
        coefficient = (normalized_rating * self.rating_weight) + \
                      (normalized_days * self.rotation_weight) - \
                      (normalized_pallets * self.distribution_weight)
                      
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
                print(f"  - {p.id}: at max capacity of {p.max_capacity}")
        
        # Keep distributing until all pallets are allocated or no eligible producers remain
        while remaining_pallets > 0 and eligible_producers:
            # Calculate coefficient for each producer
            for producer in eligible_producers:
                producer.current_coefficient = self.calculate_merit_coefficient(
                    producer, current_date, program_total_pallets
                )
            
            # Sort producers by coefficient (highest first)
            eligible_producers.sort(key=lambda p: p.current_coefficient, reverse=True)
            
            # Allocate a pallet to the producer with highest coefficient
            best_producer = eligible_producers[0]
            allocation[best_producer.id] += 1
            
            # Update the producer's delivered pallets and apply the delivery
            best_producer.delivered_in_program += 1
            
            # Remove producer if they've reached the maximum allocation or max capacity
            if (allocation[best_producer.id] >= max_pallets_per_producer or 
                not best_producer.has_capacity()):
                eligible_producers.remove(best_producer)
            
            remaining_pallets -= 1
            
            # If all producers have reached their max or capacity, report the issue
            if not eligible_producers and remaining_pallets > 0:
                print(f"Warning: All producers reached maximum allocation or capacity.")
                print(f"Unable to distribute {remaining_pallets} pallets.")
                break
        
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
            max_capacity=20,
            last_delivery_date=current_date - timedelta(days=1),
            delivered_in_program=0
        ),
        Producer(
            id='P002',
            rating=4.5,
            max_capacity=15,
            last_delivery_date=current_date - timedelta(days=5),
            delivered_in_program=0
        ),
        Producer(
            id='P003',
            rating=4,
            max_capacity=12,
            last_delivery_date=current_date - timedelta(days=3),
            delivered_in_program=0
        ),
        Producer(
            id='P004',
            rating=3,
            max_capacity=25,
            last_delivery_date=current_date - timedelta(days=30),
            delivered_in_program=0
        ),
        Producer(
            id='P005',
            rating=2,
            max_capacity=10,
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
        capacity_info = f"Capacity: {p.max_capacity}, Used: {program_deliveries}, Remaining: {p.remaining_capacity()}"
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
        capacity_status = f"Used: {program_deliveries}/{p.max_capacity} ({program_deliveries/p.max_capacity*100:.1f}%)"
        print(f"{p.id}: {program_deliveries} pallets, {capacity_status}")


# def run_example_with_tight_capacity():
#     """Run an example with tighter capacity constraints."""
#     producers, current_date = create_example_data()
#     
#     # Set tighter capacity limits
#     producers[0].max_capacity = 12  # Producer A already has 10, so only 2 more available
#     producers[1].max_capacity = 8   # Producer B already has 5, so only 3 more available
#     producers[2].max_capacity = 10  # Producer C already has 8, so only 2 more available
#     
#     print("\n\n================ Example with Tight Capacity Constraints ================")
#     print("Producer capacity limits:")
#     for p in producers:
#         program_deliveries = p.delivered_in_program
#         remaining = p.remaining_capacity()
#         print(f"{p.id}: Capacity: {p.max_capacity}, Already delivered: {program_deliveries}, Remaining: {remaining}")
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