import cirq
import numpy as np
import time
import logging
from threading import Lock
from collections import deque

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("QuantumKernel")

class TaskStatus:
    """Task states in the kernel"""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"

class QuantumTask:
    """Representation of a quantum task in the kernel"""
    
    def __init__(self, circuit, qubits, priority=0, task_id=None, metadata=None):
        self.circuit = circuit  # Quantum circuit
        self.qubits = qubits    # Allocated qubits
        self.priority = priority  # Execution priority
        self.task_id = task_id or int(time.time() * 1e6)  # Unique identifier
        self.status = TaskStatus.PENDING  # Initial state
        self.metadata = metadata or {}  # Additional metadata
        self.created_time = time.time()
        self.start_time = None
        self.end_time = None

class QuantumKernel:
    """
    Quantum Kernel for managing quantum resources and tasks
    
    Core Responsibilities:
    1. Qubit management and allocation
    2. Quantum task scheduling  
    3. Quantum circuit execution
    4. Performance monitoring and efficiency tracking
    """
    
    def __init__(self, total_qubits=10):
        # 🔧 Quantum Resources
        self.total_qubits = total_qubits
        self.available_qubits = [cirq.LineQubit(i) for i in range(total_qubits)]
        self.allocated_qubits = set()  # Currently used qubits
        
        # 📋 Task Management
        self.task_queue = deque()  # Task waiting queue
        self.completed_tasks = []  # Completed tasks
        self.failed_tasks = []     # Failed tasks
        self.task_counter = 0      # Task counter
        
        # ⚡ Simulator
        self.simulator = cirq.Simulator()
        self._lock = Lock()  # Lock for thread safety
        
        # ⚙️ Kernel Configuration
        self.config = {
            'max_concurrent_tasks': 3,
            'default_measurement_shots': 10000,
            'max_circuit_depth': 100,
            'task_timeout': 30  # seconds
        }
        
        # 📊 Performance Metrics
        self.performance_metrics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0,
            'system_efficiency': 0.0,
            'qubit_utilization': 0.0
        }
        
        logger.info(f"🔧 Quantum Kernel initialized with {total_qubits} qubits")

    def allocate_qubits(self, num_qubits):
        """
        Allocate qubits for a task
        
        Args:
            num_qubits: Number of qubits required
            
        Returns:
            list: List of allocated qubits
            
        Raises:
            ValueError: If not enough qubits available
        """
        with self._lock:
            available = self.total_qubits - len(self.allocated_qubits)
            if num_qubits > available:
                raise ValueError(f"❌ Not enough qubits. Requested: {num_qubits}, Available: {available}")
            
            # Select available qubits
            qubits = [q for q in self.available_qubits if q not in self.allocated_qubits][:num_qubits]
            
            # Allocate qubits
            for q in qubits:
                self.allocated_qubits.add(q)
            
            logger.debug(f"✅ Allocated {num_qubits} qubits: {[q.x for q in qubits]}")
            return qubits

    def release_qubits(self, qubits):
        """
        Release qubits after task completion
        
        Args:
            qubits: List of qubits to release
        """
        with self._lock:
            for q in qubits:
                self.allocated_qubits.discard(q)
            logger.debug(f"🔄 Released {len(qubits)} qubits")

    def submit_task(self, circuit, qubits, priority=0, metadata=None):
        """
        Add a new task to the waiting queue
        
        Args:
            circuit: Quantum circuit
            qubits: Allocated qubits
            priority: Execution priority
            metadata: Additional metadata
            
        Returns:
            int: Task ID
        """
        with self._lock:
            task = QuantumTask(circuit, qubits, priority, metadata=metadata)
            task.task_id = self.task_counter
            self.task_counter += 1
            
            # Add task to queue with priority consideration
            self._add_task_with_priority(task)
            
            logger.info(f"📥 Task {task.task_id} submitted - Qubits: {len(qubits)}, Depth: {len(circuit)}")
            return task.task_id

    def _add_task_with_priority(self, task):
        """Add task to queue with priority consideration"""
        if not self.task_queue:
            self.task_queue.append(task)
        else:
            # Insert task based on priority
            for i, existing_task in enumerate(self.task_queue):
                if task.priority > existing_task.priority:
                    self.task_queue.insert(i, task)
                    return
            self.task_queue.append(task)

    def execute_task(self, task_id):
        """
        Execute a specific task
        
        Args:
            task_id: Task identifier
            
        Returns:
            tuple: (Execution result, metadata)
            
        Raises:
            ValueError: If task not found
        """
        task = self._get_task_by_id(task_id)
        if not task:
            raise ValueError(f"❌ Task {task_id} not found")

        # Update task status
        task.status = TaskStatus.RUNNING
        task.start_time = time.time()

        try:
            # 🎯 Execute quantum circuit
            result = self.simulator.run(
                task.circuit, 
                repetitions=self.config['default_measurement_shots']
            )
            
            execution_time = time.time() - task.start_time
            task.end_time = time.time()

            # ✅ Update task status
            task.status = TaskStatus.COMPLETED
            self._move_task_to_completed(task)
            
            # 📊 Update metrics
            self._update_metrics(execution_time, success=True)
            
            logger.info(f"✅ Task {task_id} completed in {execution_time:.3f}s")
            
            return result, {
                'execution_time': execution_time,
                'circuit_depth': len(task.circuit),
                'qubits_used': len(task.qubits),
                'status': 'SUCCESS'
            }

        except Exception as e:
            # ❌ Handle errors
            execution_time = time.time() - task.start_time
            task.status = TaskStatus.ERROR
            self._move_task_to_failed(task)
            self._update_metrics(execution_time, success=False)
            
            logger.error(f"❌ Task {task_id} failed: {str(e)}")
            raise e
            
        finally:
            # 🔄 Release resources
            self.release_qubits(task.qubits)

    def _get_task_by_id(self, task_id):
        """Find task by ID"""
        for task in self.task_queue:
            if task.task_id == task_id:
                return task
        return None

    def _move_task_to_completed(self, task):
        """Move task to completed list"""
        if task in self.task_queue:
            self.task_queue.remove(task)
        self.completed_tasks.append(task)

    def _move_task_to_failed(self, task):
        """Move task to failed list"""
        if task in self.task_queue:
            self.task_queue.remove(task)
        self.failed_tasks.append(task)

    def _update_metrics(self, exec_time, success):
        """Update kernel performance metrics"""
        with self._lock:
            self.performance_metrics['total_tasks'] += 1
            self.performance_metrics['total_execution_time'] += exec_time
            
            if success:
                self.performance_metrics['completed_tasks'] += 1
            else:
                self.performance_metrics['failed_tasks'] += 1

            # Calculate efficiency
            total = self.performance_metrics['total_tasks']
            if total > 0:
                self.performance_metrics['average_execution_time'] = (
                    self.performance_metrics['total_execution_time'] / total
                )
                self.performance_metrics['system_efficiency'] = (
                    self.performance_metrics['completed_tasks'] / total
                )
                
            # Calculate qubit utilization
            self.performance_metrics['qubit_utilization'] = (
                len(self.allocated_qubits) / self.total_qubits
            )

    def get_system_status(self):
        """
        Get current system status
        
        Returns:
            dict: System status information
        """
        with self._lock:
            return {
                'total_qubits': self.total_qubits,
                'available_qubits': self.total_qubits - len(self.allocated_qubits),
                'allocated_qubits': len(self.allocated_qubits),
                'pending_tasks': len(self.task_queue),
                'running_tasks': len([t for t in self.task_queue if t.status == TaskStatus.RUNNING]),
                'completed_tasks': len(self.completed_tasks),
                'failed_tasks': len(self.failed_tasks),
                'performance': self.performance_metrics.copy(),
                'system_health': self._get_system_health()
            }

    def _get_system_health(self):
        """Evaluate system health"""
        efficiency = self.performance_metrics['system_efficiency']
        if efficiency >= 0.9:
            return "🟢 EXCELLENT"
        elif efficiency >= 0.7:
            return "🟡 GOOD" 
        elif efficiency >= 0.5:
            return "🟠 FAIR"
        else:
            return "🔴 NEEDS_ATTENTION"

    def clear_completed_tasks(self):
        """Clear completed tasks to free memory"""
        with self._lock:
            self.completed_tasks.clear()
            logger.info("🧹 Completed tasks cleared")

    def get_task_info(self, task_id):
        """Get information about a specific task"""
        for task in self.task_queue + self.completed_tasks + self.failed_tasks:
            if task.task_id == task_id:
                return {
                    'task_id': task.task_id,
                    'status': task.status,
                    'circuit_depth': len(task.circuit),
                    'qubits_used': len(task.qubits),
                    'created_time': task.created_time,
                    'start_time': task.start_time,
                    'end_time': task.end_time,
                    'metadata': task.metadata
                }
        return None

    def execute_immediate(self, circuit, num_qubits):
        """
        Quick execution without task management
        Useful for simple circuits
        """
        qubits = self.allocate_qubits(num_qubits)
        try:
            start_time = time.time()
            result = self.simulator.run(circuit, repetitions=self.config['default_measurement_shots'])
            execution_time = time.time() - start_time
            return result, execution_time
        finally:
            self.release_qubits(qubits)

# Example usage of the kernel
def demo_quantum_kernel():
    """Demonstration of how to use the quantum kernel"""
    
    # Create kernel
    kernel = QuantumKernel(total_qubits=5)
    
    # Create a simple quantum circuit
    qubits = kernel.allocate_qubits(2)
    circuit = cirq.Circuit()
    
    # Add quantum operations
    circuit.append(cirq.H(qubits[0]))
    circuit.append(cirq.CNOT(qubits[0], qubits[1]))
    circuit.append(cirq.measure(*qubits, key='result'))
    
    # Submit task
    task_id = kernel.submit_task(
        circuit, 
        qubits, 
        metadata={'type': 'demo', 'description': 'Bell state circuit'}
    )
    
    # Execute task
    try:
        result, metadata = kernel.execute_task(task_id)
        print(f"🎯 Task executed successfully!")
        print(f"📊 Results: {result.histogram(key='result')}")
        print(f"⏱️  Execution time: {metadata['execution_time']:.3f}s")
        
    except Exception as e:
        print(f"❌ Task failed: {e}")
    
    # Display system status
    status = kernel.get_system_status()
    print(f"\n📈 System Status:")
    print(f"   Qubits: {status['available_qubits']}/{status['total_qubits']} available")
    print(f"   Tasks: {status['pending_tasks']} pending, {status['completed_tasks']} completed")
    print(f"   Efficiency: {status['performance']['system_efficiency']:.1%}")
    print(f"   Health: {status['system_health']}")

if __name__ == "__main__":
    demo_quantum_kernel()
