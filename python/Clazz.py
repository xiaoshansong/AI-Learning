

class Clazz:
    clazz_var_1 = 0    # class variable
    clazz_var_2 = "hello"    # class variable

    def __init__(self, name):
        self.name = name    # instance variable

    def say_hello(self):
        print("Hello, my name is " + self.name)

    @classmethod
    def say_hello_class(cls):
        cls.__name__ = "Class"

    @staticmethod
    def say_hello_static():    # static method
        print("Hello, I am a static method")


# create an instance of the class
obj = Clazz("John")


# call the say_hello method
obj.say_hello()


def pre_train():
    print("Pre-training...")
    return "Pre-trained model", "Pre-trained weights", "Pre-trained hyperparameters";

def train(pre_trained_model, pre_trained_weights, pre_trained_hyperparameters):
    print("Training...")
    a,b,c = pre_train()
    print("a:",a,"b:",b,"c:",c);
    return "Trained model", "Trained weights", "Trained hyperparameters";

train("a"  , "b"  , "c"  ); # call the train method with pre-trained model, weights, and hyperparameters
