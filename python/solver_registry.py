class SolverRegistry:
    def __init__(self):
        self.solver_mapping = {}
        self.geometry_mapping = {}

    def register(self, solver_name, solver_class, geometry_class):
        self.solver_mapping[solver_name] = solver_class
        self.geometry_mapping[solver_name] = geometry_class

    def select(self, solver_name):
        if solver_name in self.solver_mapping and solver_name in self.geometry_mapping:
            solver = self.solver_mapping[solver_name]
            geometry = self.geometry_mapping[solver_name]
            return solver, geometry
        else:
            print(f"Invalid solver name: {solver_name}")
            raise ValueError

registry = SolverRegistry()