
a =  """
This is a Python program to demonstrate the use of strings in Python.

Strings are used to store and manipulate text data in Python. They are immutable sequences of Unicode characters.

In Python, strings can be enclosed in single quotes ('...') or double quotes ("...").

Here are some examples of strings in Python:
"""
len(a) # returns the length of the string
print(a.split())
print(a.ljust(1,'a'))

import io
str = io.StringIO("ddddddsdfsfwefesfsf")
print(id(str))
str.seek(3)
str.write("***")
print(str.getvalue())
print(id(str))

