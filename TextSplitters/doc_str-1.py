from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """# Define a class
class Student:
    
    # Constructor
    def __init__(self, name, age):
        self.name = name
        self.age = age

    # Method
    def introduce(self):
        print(f"My name is {self.name} and I am {self.age} years old.")

    # Method
    def study(self, subject):
        print(f"{self.name} is studying {subject}.")


# Create objects
student1 = Student("Ali", 21)
student2 = Student("Sara", 20)

# Access attributes
print(student1.name)   # Ali
print(student2.age)    # 20

# Call methods
student1.introduce()
student2.study("Python")"""

splitter = RecursiveCharacterTextSplitter.from_language(
       language = Language.PYTHON,
       chunk_size = 200,
       chunk_overlap = 0
)

chunks = splitter.split_text(text)
print(len(chunks))
print(chunks[0])
print(chunks[1])

