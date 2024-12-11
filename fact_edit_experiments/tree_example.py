class Person:
    def __init__(self, name, parents=None):
        self.name = name
        self.parents = parents if parents else []
        self.children = []
        
        for parent in self.parents:
            parent.add_child(self)

    def add_child(self, child):
        self.children.append(child)

    def get_ancestors(self, depth=0):
        ancestors = []
        for parent in self.parents:
            ancestors.append((depth, parent))
            ancestors.extend(parent.get_ancestors(depth + 1))
        return ancestors

    def get_descendants(self, depth=0):
        descendants = []
        for child in self.children:
            descendants.append((depth, child))
            descendants.extend(child.get_descendants(depth + 1))
        return descendants
    
    def print_family_tree(self):
        for child in self.children:
            print(f"{child.name} is a child of {self.name}")
            # also print their parents
            for parent in child.parents:
                if parent != self:
                    print(f"{child.name} is a child of {parent.name}")
            child.print_family_tree()


# Example usage:
if __name__ == "__main__":
    # Create individuals
    john = Person("John")
    mary = Person("Mary")

    alice = Person("Alice", parents=[john, mary])
    bob = Person("Bob", parents=[john, mary])

    carol = Person("Carol", parents=[alice])
    dave = Person("Dave", parents=[alice, bob])

    emily = Person("Emily", parents=[carol])
    frank = Person("Frank", parents=[carol])

    dave = Person("Dave", parents=[alice, bob])

    jessy = Person("Jessy", parents=[dave])
    kate = Person("Kate", parents=[dave])
    penny = Person("Penny", parents=[dave])

    john.print_family_tree()

    # Make another similar one with different than above
    perry = Person("Perry")
    quincy = Person("Quincy")

    rachel = Person("Rachel", parents=[perry, quincy])
    sam = Person("Sam", parents=[perry, quincy])
    jocelyn = Person("Jocelyn", parents=[rachel])
    kyle = Person("Kyle", parents=[rachel, sam])
    luke = Person("Luke", parents=[rachel, sam])
    molly = Person("Molly", parents=[rachel, sam])
    nancy = Person("Nancy", parents=[rachel, sam])

    perry.print_family_tree()

    