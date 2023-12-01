# Zyph

Zyph is a python interpreted programming language. These are its features!
When you've finished reading through all the features, go to the bottom of the page to learn how to install Zyph.

## Variables

Variables in Zyph can be declared using the following syntax:

```
var name = value
```

A variable can be everything in the language, functions too.

```
print("foo")

var printVar = print

printVar("foo")
```

The sintax for when a variable is reasigned is always the same.

```
var foo = 'bar'

var foo = 'faz'
```

## Functions

A function in Zyph is declared using the following syntax:

```
function name(args) {
    body
}
```

Functions can also return values by using the return keyword:

```
function add(n1, n2) {
    return n1 + n2
}
```

## Lists

Lists in Zyph are created with square brackets and can contain any type of data including other lists.

```
var list = ["foo", "bar", ["foo", "bar"]]
```

You can access elements in a list using the following syntax:

```
var list = [1, 2, 3]

print(list / 0) // prints 1
```

To concatenate two lists together you use the asterisk:

```
var listOne = [1, 2, 3]
var listTwo = [4, 5, 6]


print(listOne * tistTwo) // prints [1, 2, 4, 5, 6]
```

To append or remove an element from a list you use the plus and minus signs respectively:

```
var list = [1, 2, 3]

list + 4
print(list) // prints [1, 2, 3, 4]

list - 2
print(list) // prints [1, 2, 4]
```

When you remove an element you are removing it by its index.

## Booleans
Booleans represent true or false values. They are used to control flow within your code.
Zyph dosn't have true of false values but they get rappresented by 1 (true) and 0 (false).
If you want to use true and false there are two constants in the language.

```
print(true) // prints 1

print(false) // prints 0
```

To check differences between two values you can use the operators in the language:

```
== // equal to
!= // not equal to
<= // equal to or smaller
>= // equal to or bigger
<  // smaller than
>  // bigger than
```

## Values

In Zyph the avaliable values are strings, numbers (intigers or floats), lists and functions.
I'm working to add other values like classes or objects to the language but for now this is all i got.

## Built in functions

There are some built-in functions that you can use in zyph. Here is a list with them:

```
print()
print_ret()
input()
input_int()
clear()
cls()
is_num()
is_str()
is_list()
is_function()
append()
pop()
extend()
sort()
len()
count() // currently works only in strings
str()
num()
int()
float()
abs()
min()
max()
range()
factorial()
raise_error()
randfloat()
randint()
run()
quit()
```

There are also some constants in the language, like the previously mentioned 'true' and 'false' constants and:

```
math_pi  // approximation of pi
math_e   // approximation of euler number
null
```

## Statements

Statements are used to control the flow of your program. There are several types of statements available in Zyph. Here is a list with them and their syntax:

### if, elif and else statements

```
if (condition) {
    if-body
} elif (condition) {
    elif-body
} else {
    else-body
}
```

Due to my limited coding skills as right now the only way to rappresent if, elif and else or any other multi keyword statement is like this but i'm working on fixing it.

### for and while loops

```
for (index = starting-position, to ending-position, step step-value) {
    body
}
```
The step value is option and, by default, will be 1.

```
while (condition) {
    body
}
```

The result of for and while loops will always be in a list

### try and catch statements

```
try {
    try-body
} catch {
    catch-body
}
```
As right now you can't specify the time of exception you are tring to catch but i'm currently working on it.

### continue, break and return

```
for(i = 0, to 100) {
    var i = i + 1
    if (i == 50) {
        continue
    } elif (i == 70) {
        break
    }
}



function sum(n1, n2) {
    return n1 + n2
}
```

## Newlines and semicolons
In this language newlines are used as separators between commands or expressions. You don't need to use semicolon at the end of a command as they are only used for single line statements but if you want you can use them

```
var x = 3;
if (x > 4) {
    print("true");
} else {
    print("false");
};
```

# Installation

Clone the repository and run the shell.py file for the terminal interpreter, if you want to write your code in a file create one ending in .zys and write your code, then you can run it by writing in your terminal:

```
python shell.py your/file/directory
```

or by wrinting in the terminal interpreter

```
run('your/file/directory')
```

## Vs Code syntax highlighting

Get in the cloned repository and move the 'zyph' folder in your Vs Code extensions direcory (in linux usually ~/.vscode/extensions) or running the command:

```
cp -r zyph/ ~/.vscode/extensions
```

to automatically move the folder.
Then restart Vs Code or open the Command Palette pressing <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>P</kbd> and write "Developer: Reload Window".