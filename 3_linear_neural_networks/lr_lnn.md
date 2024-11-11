# Linear Regression & Linear Neural Networks

## Basics

- We assume that the relationship between features $\mathbf{x}$ and target $y$ is approximately linear, i.e. $E[Y|X=\mathbf{x}]$ can be expressed as a weighted sum of the features $\mathbf{x}$.
- Collecting all features into a vector $\mathbf{x} \in \mathbb{R}^d$ and all weights into a vector $\mathbf{w} \in \mathbb{R}^d$, we can write the linear model as

$$
\hat{y} = \mathbf{w}^T \mathbf{x} + b
$$

- We often find it convenient to refer to features of our entire dataset of $n$ examples via the *design matrix* $\mathbf{X} \in \mathbb{R}^{n \times d}$, where each row is an example and each column is a feature. The predictions $\hat{\mathbf{y}}$ can be written as

$$
\hat{\mathbf{y}} = \mathbf{X} \mathbf{w} + b
$$

- **Loss function**: squared error function:

$$
l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \sum_{i=1}^n (\hat{y}^{(i)} - y^{(i)})^2
$$

    In quadratic form:

$$
L(\mathbf{w}, b) = \frac{1}{2} \sum_{i=1}^n (\mathbf{w}^T \mathbf{x}^{(i)} + b - y^{(i)})^2
$$

- **Objective**: minimize the loss function over the entire dataset:

$$
\mathbf{w}^*, b^* = \arg \min_{\mathbf{w}, b} L(\mathbf{w}, b)
$$

- **Analytical solution**:

$$
\mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

## Gradient Descent

- **Batch Gradient Descent**:

  - Update rule:

  $$
  begin{aligned}
  \mathbf{w} &\leftarrow \mathbf{w} - \eta \nabla_{\mathbf{w}} L(\mathbf{w}, b) \\
  b &\leftarrow b - \eta \nabla_b L(\mathbf{w}, b)
  \end{aligned}
  $$

  - Gradient:

  $$
  begin{aligned}
  \nabla_{\mathbf{w}} L(\mathbf{w}, b) &= \mathbf{X}^T (\mathbf{X} \mathbf{w} + b - \mathbf{y}) \\
  \nabla_b L(\mathbf{w}, b) &= \sum_{i=1}^n (\mathbf{w}^T \mathbf{x}^{(i)} + b - y^{(i)})
  \end{aligned}
  $$
- **Stochastic Gradient Descent**:

  $$
  begin{aligned}
  \mathbf{w} &\leftarrow \mathbf{w} - \eta \nabla_{\mathbf{w}} l^{(i)}(\mathbf{w}, b) \\
  b &\leftarrow b - \eta \nabla_b l^{(i)}(\mathbf{w}, b)
  \end{aligned}
  $$
- **Mini-batch Gradient Descent**:

  $$
  begin{aligned}
  \mathbf{w} &\leftarrow \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \nabla_{\mathbf{w}} l^{(i)}(\mathbf{w}, b) \\
  b &\leftarrow b - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \nabla_b l^{(i)}(\mathbf{w}, b)
  \end{aligned}
  $$

## Linear Regression as a Neural Network

<img src="https://www.d2l.ai/_images/singleneuron.svg" alt="../_images/singleneuron.svg" style="zoom:100%;" />







# Coding Notes

## Usage of `@` in Python

在 Python 中，`@` 符号通常用于**装饰器**和**矩阵运算**。以下是它在不同场景下的用法和详细解释：

### 1. 装饰器中的 `@` 用法

`@` 符号最常用于装饰器。装饰器是一种特殊的函数，可以用来修改其他函数或类的行为。它的语法是将 `@decorator_name` 放在函数或方法的定义之前。

#### 基本用法

```python
@decorator_name
def some_function():
    pass
```

等同于

```python
def some_function():
    pass
some_function = decorator_name(some_function)
```

通过装饰器可以在不修改函数定义的情况下增强函数功能。这对代码的复用性、可读性和结构化非常有帮助。

#### 示例：装饰器用于函数

```python
def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
```

输出：
```
Something is happening before the function is called.
Hello!
Something is happening after the function is called.
```

#### 装饰器用于类

装饰器也可以用于类，来修改类的行为或添加额外的功能。

```python
def add_attributes(cls):
    cls.new_attribute = "I am a new attribute"
    return cls

@add_attributes
class MyClass:
    pass

print(MyClass.new_attribute)  # 输出 "I am a new attribute"
```

### 2. `@` 符号用于矩阵运算

Python 3.5 及以后版本中引入了 `@` 运算符用于矩阵乘法，主要是为 `NumPy` 或其他矩阵库设计的。

#### 示例：矩阵运算

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 使用 @ 进行矩阵乘法
C = A @ B
print(C)
```

输出：
```
[[19 22]
 [43 50]]
```

### 3. 常见的内置装饰器

Python 中有一些常用的内置装饰器，如 `@property`, `@staticmethod`, `@classmethod` 等，这些装饰器可以直接作用于类的方法。

#### `@property`

将类的方法转换为属性，使得我们可以像访问属性一样访问方法。

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        return self._radius

c = Circle(5)
print(c.radius)  # 直接访问属性，不需要调用方法
```

#### `@staticmethod` 和 `@classmethod`

- `@staticmethod` 装饰器用于将方法定义为静态方法，可以直接通过类调用而不依赖于实例。
- `@classmethod` 装饰器将方法定义为类方法，第一个参数接收的是类本身，而不是实例。

```python
class MyClass:
    @staticmethod
    def static_method():
        print("This is a static method.")

    @classmethod
    def class_method(cls):
        print("This is a class method.")

MyClass.static_method()
MyClass.class_method()
```

### 总结

- **装饰器**：用 `@decorator_name` 修饰函数或类来增强功能。
- **矩阵运算**：在矩阵库中使用 `@` 进行矩阵乘法。
- **内置装饰器**：`@property`, `@staticmethod`, `@classmethod` 等，用于简化属性访问、定义静态方法和类方法等。

`@` 符号在 Python 中的多样用途，使得代码更加简洁和模块化，尤其是在函数和类的设计中非常有用。

### `add_to_class(Class)`

```python
def add_to_class(Class):  #@save
    """Register functions as methods in created class."""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper
```

#### Usage

```python
class A:
    def __init__(self):
        self.b = 1

a = A()

@add_to_class(A)
def do(self):
    print('Class attribute "b" is', self.b)

# Here we add function do() in class A
a.do()
```

## Class Inheritance in Python

### `super()`

1. **调用父类的初始化方法**：子类的 `__init__` 方法通常会调用 `super().__init__()`，以确保父类的初始化逻辑被执行。这在多重继承和复杂类层次结构中尤为重要。
2. **访问父类的其他方法**：除了初始化方法 `__init__`，`super()` 还可以用于调用父类的任何方法。
3. **支持多重继承**：在多重继承的情况下，`super()` 会按照方法解析顺序（Method Resolution Order, MRO）依次调用父类的方法，以确保所有父类的方法被正确执行。

#### 示例代码

假设我们有一个父类 `Animal` 和一个子类 `Dog`，子类 `Dog` 继承了 `Animal` 的属性和方法：

```python
pythonCopy codeclass Animal:
    def __init__(self, name):
        self.name = name

    def make_sound(self):
        print("Some generic sound")

class Dog(Animal):
    def __init__(self, name, breed):
        # 调用父类的 __init__ 方法
        super().__init__(name)
        self.breed = breed

    def make_sound(self):
        # 调用父类的 make_sound 方法
        super().make_sound()
        print("Bark!")

# 使用子类
dog = Dog("Buddy", "Golden Retriever")
dog.make_sound()
```

#### 解释

1. `super().__init__(name)` 调用了 `Animal` 的初始化方法，设置了 `self.name` 属性。
2. `super().make_sound()` 调用了父类 `Animal` 的 `make_sound` 方法，打印出 "Some generic sound"。
3. 随后，`Dog` 类的 `make_sound` 方法打印出 "Bark!"，展示了 `super()` 调用父类方法的同时允许子类添加自己的行为。



## Usage of `yield`

在Python中，`yield` 是一个关键字，用来生成一个迭代器。它的作用类似于 `return`，不过它不会终止函数，而是“暂停”函数的执行，将值返回给调用者。下次再次调用时，函数会从暂停的地方继续执行。

`yield` 主要用于创建生成器（generator），它是一种特殊的迭代器，允许在每次迭代时生成一个值，而不需要一次性把所有值存储在内存中。下面是如何使用 `yield` 的基本示例和一些实际应用。

### 基本示例

```python
def count_up_to(n):
    count = 1
    while count <= n:
        yield count  # 暂停并返回 count 值
        count += 1   # 下一次调用时继续执行

# 使用生成器
counter = count_up_to(5)
for number in counter:
    print(number)
```

输出：
```
1
2
3
4
5
```

在这里，`count_up_to` 是一个生成器函数，每次循环中遇到 `yield` 时，它返回当前的 `count` 值并暂停执行。调用者可以继续请求下一个值，而生成器会从暂停的地方继续执行。

> `counter` 不是一个 `list`，而是一个 **生成器对象**（generator object）。
>
> 生成器对象类似于一个迭代器，可以用 `for` 循环或 `next()` 函数来逐个获取值，但它不会一次性生成所有值并存储在内存中，而是按需生成每个值，这也是生成器的主要优势。
>
> 我们可以通过以下方式验证 `counter` 不是一个 `list`：

### `yield` 与 `return` 的不同

- `yield` 暂停函数，并允许它从暂停的地方继续执行，而 `return` 会直接结束函数。
- 一个包含 `yield` 的函数是生成器函数，返回的是一个生成器对象，而不是单个值。

### 实际应用示例

1. **生成无限序列**（如斐波那契数列）

   ```python
   def fibonacci():
       a, b = 0, 1
       while True:
           yield a
           a, b = b, a + b

   fib = fibonacci()
   for _ in range(10):
       print(next(fib))
   ```

   这会生成前 10 个斐波那契数。因为生成器函数是惰性求值的，每次 `next` 调用时才生成下一个数，内存占用小。

2. **分块处理大数据文件**

   ```python
   def read_large_file(file_path, chunk_size=1024):
       with open(file_path, 'r') as file:
           while True:
               data = file.read(chunk_size)
               if not data:
                   break
               yield data

   # 使用生成器逐块读取
   for chunk in read_large_file('large_file.txt'):
       process_data(chunk)
   ```

   这种方法可以在读取超大文件时节省内存，因为它每次只读取一小块数据而不是一次性加载全部内容。

3. **生成器表达式**（类似于列表推导式）

   ```python
   squares = (x * x for x in range(10))
   for square in squares:
       print(square)
   ```

   生成器表达式与列表推导式类似，但不会立即计算所有结果，而是按需生成元素。

### 总结

- `yield` 用于构建生成器函数，通过暂停函数来返回值。
- 生成器可以节省内存，适合处理大量数据或无限序列。
- 使用 `yield` 时，生成器函数会返回一个生成器对象，可以用 `for` 循环或 `next()` 函数来遍历生成的值。
