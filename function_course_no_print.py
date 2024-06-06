from prettytable import PrettyTable
from sympy import *
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def call_counter(func):
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return func(*args, **kwargs)
    wrapper.calls = 0
    return wrapper

@call_counter
def f(x, y = None): 
    if y == None:
        x1, x2 = x
    else:
        x1, x2 = x, y
    # return pow(x1,2) + pow(x2,2)
    return pow((10*pow((x1-x2),2) + pow((x1-1),2)),0.25)


def add(point_1, point_2):
    x1, y1 = point_1
    x2, y2 = point_2
    return ((x1 + x2), (y1 + y2))   

def mult(multiplier, point):
    x1, x2 = point
    return (multiplier * x1, multiplier * x2)

def div(point_1, point_2):
    x1, y1 = point_1
    if isinstance(point_2, tuple):
        x2, y2 = point_2
        return ((x1 / x2), (y1 / y2))
    else:
        return ((x1 / point_2), (y1 / point_2))

def norm(point):
    x1, x2 = point
    return pow((pow(x1,2) + pow(x2,2)),0.5)

def grad_c(point, h=0.00001):
    x1, x2 = point
    dx = (f(x1 + h, x2) - f(x1 - h, x2)) / (2 * h)
    dy = (f(x1, x2 + h) - f(x1, x2 - h)) / (2 * h)
    return (dx, dy)

def grad_pr(point, h=0.00001):
    x1, x2 = point
    dx = (f(x1 + h, x2) - f(x1, x2)) / (h)
    dy = (f(x1, x2 + h) - f(x1, x2)) / (h)
    return (dx, dy)

def grad_l(point, h=0.00001):
    x1, x2 = point
    dx = (f(x1, x2) - f(x1 - h, x2)) / (h)
    dy = (f(x1, x2) - f(x1, x2 - h)) / (h)
    return (dx, dy)

def svenns_algorithm(point, way, lambda0=0, lambda_step_multiplier=0.001):
    lambda_step = lambda_step_multiplier*((pow((point[0]*point[0] + point[1]*point[1]),0.5))/(pow((way[0]*way[0] + way[1]*way[1]),0.5)))
    func_lambda0 = f(add(point, mult(lambda0, way)))
    added = lambda0 + lambda_step
    subtracted = lambda0 - lambda_step
    func_added = f(add(point, mult(added, way)))
    func_subtracted= f(add(point, mult(subtracted, way)))
    table = PrettyTable()
    table.clear()
    table.field_names = ["","%lambda","x", "y", "f(%lambda)"]
    table.add_row(["%lambda _0",lambda0, point[0], point[1], f(add(point, mult(lambda0, way)))])
    added_point = add(point, mult(added, way))
    table.add_row(["%lambda _0 + %DELTA %lambda",added, added_point[0], added_point[1], func_added])
    subtracted_point = add(point, mult(subtracted, way))
    table.add_row(["%lambda _0 - %DELTA %lambda",subtracted, subtracted_point[0], subtracted_point[1], func_subtracted])
    if func_lambda0 < func_added and func_lambda0 < func_subtracted:
        a,b = subtracted, added
        print(f"Отже, інтервал невизначеності : [{a},{b}]")
        return a,b
    else:
        if func_added < func_subtracted:
            deltalambda = added
            temp_lambda = deltalambda
            i=1
            while (1+1==2):
                func_previous_lambda = f(add(point, mult(temp_lambda, way)))
                new_lambda = temp_lambda + lambda_step*pow(2,i)
                func_new_lambda = f(add(point, mult(new_lambda, way)))
                new_point = add(point, mult(new_lambda, way))
                table.add_row([f"%lambda _{i+1}",new_lambda, new_point[0], new_point[1], func_new_lambda])
                if func_new_lambda > func_previous_lambda:
                    break
                temp_lambda = new_lambda
                i+=1
                func_previous_lambda = f(add(point, mult(temp_lambda, way)))
            mean_lambda =  (temp_lambda + new_lambda)/2
            func_mean_lambda = f(add(point, mult(mean_lambda, way)))
            mean_point = add(point, mult(mean_lambda, way))
            table.add_row([f"%lambda _{i+2}",mean_lambda, mean_point[0], mean_point[1], func_mean_lambda])
            with open('temp.csv', 'w', newline='') as f_output:
                f_output.write(table.get_csv_string())
            data = pd.read_csv("temp.csv")
            sorted_df = data.sort_values(by='f(%lambda)')
            next_to_smallest = sorted_df.iloc[1:3]
            next_to_smallest_sorted = next_to_smallest.sort_values(by='f(%lambda)')
            lambdas = next_to_smallest_sorted['%lambda'].tolist()
            os.remove("temp.csv")
            print(f"Отже, інтервал невизначеності: [{lambdas[1]},{lambdas[0]}]")
            return lambdas[0], lambdas[1]
        else:
            deltalambda = subtracted
            temp_lambda = deltalambda
            i=1
            while (1+1==2):
                func_previous_lambda = f(add(point, mult(temp_lambda, way)))
                new_lambda = temp_lambda - lambda_step*pow(2,i)
                func_new_lambda = f(add(point, mult(new_lambda, way)))
                new_point = add(point, mult(new_lambda, way))
                table.add_row([f"%lambda _{i+1}",new_lambda, new_point[0], new_point[1], func_new_lambda])
                if func_new_lambda > func_previous_lambda:
                    break
                temp_lambda = new_lambda
                i+=1
            func_previous_lambda = f(add(point, mult(temp_lambda, way)))
            mean_lambda =  (temp_lambda + new_lambda)/2
            func_mean_lambda = f(add(point, mult(mean_lambda, way)))
            mean_point = add(point, mult(mean_lambda, way))
            table.add_row([f"%lambda _{i+2}",mean_lambda, mean_point[0], mean_point[1], func_mean_lambda])
            with open('temp.csv', 'w', newline='') as f_output:
                f_output.write(table.get_csv_string())
            data = pd.read_csv("temp.csv")
            sorted_df = data.sort_values(by='f(%lambda)')
            next_to_smallest = sorted_df.iloc[1:3]
            next_to_smallest_sorted = next_to_smallest.sort_values(by='f(%lambda)')
            lambdas = next_to_smallest_sorted['%lambda'].tolist()
            os.remove("temp.csv")
            print(f"Отже, інтервал невизначеності: [{lambdas[0]},{lambdas[1]}]")
            return lambdas[0], lambdas[1]
        

def golden_ratio(point, s, a_given, b_given, e=0.01):
    a = min(a_given, b_given)
    b = max(a_given, b_given)
    L = b - a
    while L > e:
        lambda1 = a + 0.382 * L
        lambda2 = a + 0.618 * L
        if f(add(point, mult(lambda1, s))) > f(add(point, mult(lambda2, s))):
            a = lambda1
        elif f(add(point, mult(lambda1, s))) < f(add(point, mult(lambda2, s))):
            b = lambda2
        L = b - a

    print(f'Відповідь (золотий переріз): [{a}, {b}]')
    print(f'Отже візьмемо %lambda = frac(a+b)(2) = {(a+b)/2}')
    return (a+b)/2

def dsc_powell(point,s,a_given,b_given,e=0.01):
    a = min(a_given, b_given)
    b = max(a_given, b_given)
    lambda1 = a
    lambda3 = b
    lambda2 = (a+b)/2
    delta = (lambda2-lambda1)
    lambdaz = (lambda2 + (delta*((f(add(point,mult(lambda1, s)))) - (f(add(point,mult(lambda3, s))))))/(2*((f(add(point,mult(lambda1, s))))-2*(f(add(point,mult(lambda2, s))))+ (f(add(point,mult(lambda3, s)))))))
    while abs(lambda2-lambdaz)>=e or abs(((f(add(point,mult(lambda2, s))))-(f(add(point,mult(lambdaz, s))))))>=e :
        list = [lambda1, lambda2, lambda3,lambdaz]
        list.sort()
        list_with_f = []
        for i in range(len(list)):
            list_with_f.append((f(add(point,(mult(list[i], s))))))
        idx_min = list_with_f.index(min(list_with_f))
        lambda2 = list[idx_min]
        lambda1 = list[idx_min - 1]
        lambda3 = list[idx_min + 1]
        a1 = (((f(add(point,mult(lambda2, s))))) - ((f(add(point,mult(lambda1, s)))))) / (lambda2 - lambda1)
        a2 = (1/(lambda3-lambda2)) * ((((((f(add(point,mult(lambda3, s))))) - ((f(add(point,mult(lambda1, s))))))/(lambda3-lambda1))-(((f(add(point,mult(lambda2, s))))) - ((f(add(point,mult(lambda1, s))))))/(lambda2 - lambda1)))
        lambdaz = (((lambda1 + lambda2) / 2) - (a1) / (2 * (a2)))       
    print(f'Відповідь: %lambda "*"={lambdaz}')
    return lambdaz

def mns_pr_norm(point, lim=20, epsilon=0.01, dsc=1):
    i=0
    point_history = []
    while norm(grad_pr(point)) > epsilon or i < lim:
        point_history.append(point)
        prev_x = point
        S = mult(-1, grad_pr(point))
        a1,b1 = svenns_algorithm(point, S)
        if dsc==1:
            LAMBDA = dsc_powell(point, S, a1, b1)
        else:
            LAMBDA = golden_ratio(point, S, a1, b1)
        point = add(prev_x, mult(LAMBDA, S))
        print(f'x^({i+1}) = {point}^T')
        if norm(grad_pr(point)) < epsilon:
            break
        point_history.append(point)
        if i >= lim - 1:
            break
        i += 1
    x_vals = [point[0] for point in point_history]
    y_vals = [point[1] for point in point_history]

    x = np.linspace(min(x_vals) - 1, max(x_vals) + 1, 100)
    y = np.linspace(min(y_vals) - 1, max(y_vals) + 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = f((X, Y))

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=50)
    plt.plot(x_vals, y_vals, marker='o', color='r', label='Optimization Path')
    plt.colorbar()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Contour Plot with Optimization Path')
    plt.legend()
    plt.grid(True)
    plt.show()
    return None

def mns_c_norm(point, lim=20, epsilon=0.01, dsc=1):
    i=0
    point_history = []
    while norm(grad_c(point)) < epsilon or i < lim:
        point_history.append(point)
        prev_x = point
        S = mult(-1, grad_c(point))
        a1,b1 = svenns_algorithm(point, S)
        if dsc==1:
            LAMBDA = dsc_powell(point, S, a1, b1)
        else:
            LAMBDA = golden_ratio(point, S, a1, b1)
        point = add(prev_x, mult(LAMBDA, S))
        print(f'x^({i+1}) = {point}^T')
        if norm(grad_c(point)) < epsilon:
            break
        point_history.append(point)
        if i >= lim - 1:
            break
        i += 1
    x_vals = [point[0] for point in point_history]
    y_vals = [point[1] for point in point_history]

    x = np.linspace(min(x_vals) - 1, max(x_vals) + 1, 100)
    y = np.linspace(min(y_vals) - 1, max(y_vals) + 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = f((X, Y))

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=50)
    plt.plot(x_vals, y_vals, marker='o', color='r', label='Optimization Path')
    plt.colorbar()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Contour Plot with Optimization Path')
    plt.legend()
    plt.grid(True)
    plt.show()
    return None


def mns_l_norm(point, lim=20, epsilon=0.01, dsc=1):
    i=0
    point_history = []
    while i < lim:
        point_history.append(point)
        prev_x = point
        S = mult(-1, grad_l(point))
        a1,b1 = svenns_algorithm(point, S)
        if dsc==1:
            LAMBDA = dsc_powell(point, S, a1, b1)
        else:
            LAMBDA = golden_ratio(point, S, a1, b1)
        point = add(prev_x, mult(LAMBDA, S))
        print(f'x^({i+1}) = {point}^T')
        if norm(grad_l(point)) < epsilon:
            break
        point_history.append(point)
        if i >= lim - 1:
            break
        i += 1
    x_vals = [point[0] for point in point_history]
    y_vals = [point[1] for point in point_history]

    x = np.linspace(min(x_vals) - 1, max(x_vals) + 1, 100)
    y = np.linspace(min(y_vals) - 1, max(y_vals) + 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = f((X, Y))

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=50)
    plt.plot(x_vals, y_vals, marker='o', color='r', label='Optimization Path')
    plt.colorbar()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Contour Plot with Optimization Path')
    plt.legend()
    plt.grid(True)
    plt.show()
    return None

def partan_c_norm(point,epsilon=0.00001,lim=20, dsc=1):
    i=0
    point_history = []
    while norm(grad_c(point)) > epsilon:
        point_history.append(point)
        prev_x = point
        S = mult(-1, grad_c(point))
        a1,b1 = svenns_algorithm(point, S, lambda0=0)
        if dsc==1:
            LAMBDA = dsc_powell(point, S, a1, b1, e=0.0001)
        else:
            LAMBDA = golden_ratio(point, S, a1, b1)
        point = add(prev_x, mult(LAMBDA, S))
        print(f'x^({i+1}) = {point}^T')
        if norm(grad_c(point)) < epsilon:
            break
        point_history.append(point)
        prev_x_1 = point
        S_1 = mult(-1, grad_c(point))
        a2,b2 = svenns_algorithm(point, S_1,lambda0=0)
        if dsc==1:
            LAMBDA = dsc_powell(point, S_1, a2, b2, e=0.0001)
        else:
            LAMBDA = golden_ratio(point, S_1, a2, b2)
        point = add(prev_x_1, mult(LAMBDA, S_1))
        print(f'x^({i+2}) = {point}^T')
        if norm(grad_c(point)) < epsilon:
            break
        point_history.append(point)
        prev_x_2 = point
        S_2 = add(prev_x_1, mult(-1,prev_x))
        a3,b3 = svenns_algorithm(point, S_2)
        if dsc==1:
            LAMBDA = dsc_powell(point, S_2, a3, b3, e=0.0001)
        else:
            LAMBDA = golden_ratio(point, S_2, a3, b3)
        point = add(prev_x_2, mult(LAMBDA, S_2))
        print(f'x^({i+3}) = {point}^T')
        if norm(grad_c(point)) < epsilon:
            break
        point_history.append(point)
        if i >= lim:
            break
        i += 3
    x_vals = [point[0] for point in point_history]
    y_vals = [point[1] for point in point_history]

    x = np.linspace(min(x_vals) - 1, max(x_vals) + 1, 100)
    y = np.linspace(min(y_vals) - 1, max(y_vals) + 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = f((X, Y))

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=50)
    plt.plot(x_vals, y_vals, marker='o', color='r', label='Optimization Path')
    plt.colorbar()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Contour Plot with Optimization Path')
    plt.legend()
    plt.grid(True)
    plt.show()

def partan_pr_norm(point,epsilon=0.00001,lim=20, dsc=1):
    i=0
    point_history = []
    while norm(grad_pr(point)) > epsilon:
        point_history.append(point)
        prev_x = point
        S = mult(-1, grad_pr(point))
        a1,b1 = svenns_algorithm(point, S)
        if dsc==1:
            LAMBDA = dsc_powell(point, S, a1, b1)
        else:
            LAMBDA = golden_ratio(point, S, a1, b1)
        point = add(prev_x, mult(LAMBDA, S))
        print(f'x^({i+1}) = {point}^T')
        if norm(grad_pr(point)) < epsilon:
            break
        point_history.append(point)
        prev_x_1 = point
        S_1 = mult(-1, grad_pr(point))
        a2,b2 = svenns_algorithm(point, S_1)
        if dsc==1:
            LAMBDA = dsc_powell(point, S_1, a2, b2)
        else:
            LAMBDA = golden_ratio(point, S_1, a2, b2)
        point = add(prev_x_1, mult(LAMBDA, S_1))
        print(f'x^({i+2}) = {point}^T')
        if norm(grad_pr(point)) < epsilon:
            break
        point_history.append(point)
        prev_x_2 = point
        S_2 = add(prev_x_1, mult(-1,prev_x))
        a3,b3 = svenns_algorithm(point, S_2)
        if dsc==1:
            LAMBDA = dsc_powell(point, S_2, a3, b3)
        else:
            LAMBDA = golden_ratio(point, S_2, a3, b3)
        point = add(prev_x_2, mult(LAMBDA, S_2))
        print(f'x^({i+3}) = {point}^T')
        if norm(grad_pr(point)) < epsilon:
            break
        point_history.append(point)
        if i >= lim :
            break
        i += 3
    x_vals = [point[0] for point in point_history]
    y_vals = [point[1] for point in point_history]

    x = np.linspace(min(x_vals) - 1, max(x_vals) + 1, 100)
    y = np.linspace(min(y_vals) - 1, max(y_vals) + 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = f((X, Y))

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=50)
    plt.plot(x_vals, y_vals, marker='o', color='r', label='Optimization Path')
    plt.colorbar()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Contour Plot with Optimization Path')
    plt.legend()
    plt.grid(True)
    plt.show()

def partan_l_norm(point,epsilon=0.00001,lim=20, dsc=1):
    i=0
    point_history = []
    while norm(grad_l(point)) > epsilon:
        point_history.append(point)
        prev_x = point
        S = mult(-1, grad_l(point))
        a1,b1 = svenns_algorithm(point, S)
        if dsc==1:
            LAMBDA = dsc_powell(point, S, a1, b1)
        else:
            LAMBDA = golden_ratio(point, S, a1, b1)
        point = add(prev_x, mult(LAMBDA, S))
        print(f'x^({i+1}) = {point}^T')
        if norm(grad_l(point)) < epsilon:
            break 
        point_history.append(point)
        prev_x_1 = point
        S_1 = mult(-1, grad_l(point))
        a2,b2 = svenns_algorithm(point, S_1)
        if dsc==1:
            LAMBDA = dsc_powell(point, S_1, a2, b2)
        else:
            LAMBDA = golden_ratio(point, S_1, a2, b2)
        point = add(prev_x_1, mult(LAMBDA, S_1))
        print(f'x^({i+2}) = {point}^T')
        if norm(grad_l(point)) < epsilon:
            break 
        point_history.append(point)
        prev_x_2 = point
        S_2 = add(prev_x_1, mult(-1,prev_x))
        a3,b3 = svenns_algorithm(point, S_2)
        if dsc==1:
            LAMBDA = dsc_powell(point, S_2, a3, b3)
        else:
            LAMBDA = golden_ratio(point, S_2, a3, b3)
        point = add(prev_x_2, mult(LAMBDA, S_2))
        print(f'x^({i+3}) = {point}^T')
        if norm(grad_l(point)) < epsilon:
            break
        point_history.append(point)
        if i >= lim:
            break
        i += 3
    x_vals = [point[0] for point in point_history]
    y_vals = [point[1] for point in point_history]

    x = np.linspace(min(x_vals) - 1, max(x_vals) + 1, 100)
    y = np.linspace(min(y_vals) - 1, max(y_vals) + 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = f((X, Y))

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=50)
    plt.plot(x_vals, y_vals, marker='o', color='r', label='Optimization Path')
    plt.colorbar()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Contour Plot with Optimization Path')
    plt.legend()
    plt.grid(True)
    plt.show()

def partan_mod_pr_norm(point,epsilon=0.00001,lim=20, dsc=1):
    i=0
    point_history = []
    point_history.append(point)
    prev_x_0 = point
    S = mult(-1, grad_pr(point))
    a1,b1 = svenns_algorithm(point, S)
    if dsc==1:
        LAMBDA = dsc_powell(point, S, a1, b1)
    else:
        LAMBDA = golden_ratio(point, S, a1, b1)
    point = add(prev_x_0, mult(LAMBDA, S))
    print(f'x^({i+1}) = {point}^T')
    if norm(grad_pr(point)) < epsilon:
        return None
    point_history.append(point)
    prev_x_1 = point
    S_1 = mult(-1, grad_pr(point))
    a2,b2 = svenns_algorithm(point, S_1)
    if dsc==1:
        LAMBDA = dsc_powell(point, S_1, a2, b2)
    else:
        LAMBDA = golden_ratio(point, S_1, a2, b2)
    point = add(prev_x_1, mult(LAMBDA, S_1))
    print(f'x^({i+2}) = {point}^T')
    if norm(grad_pr(point)) < epsilon:
        return None 
    point_history.append(point)
    prev_x_2 = point
    i += 2
    while norm(grad_pr(prev_x_2)) > epsilon:
        S_2 = add(prev_x_2, mult(-1, prev_x_0))
        a3,b3 = svenns_algorithm(point, S_2)
        if dsc==1:
            LAMBDA = dsc_powell(point, S_2, a3, b3)
        else:
            LAMBDA = golden_ratio(point, S_2, a3, b3)
        point = add(prev_x_2, mult(LAMBDA, S_2))
        point_history.append(point)
        prev_x_3 = point
        print(f'x^({i+1}) = {point}^T')
        if norm(grad_pr(point)) < epsilon:
            return None
        i += 1
        if norm(grad_pr(point)) <= epsilon:
            break
        S = mult(-1, grad_pr(point))
        a1,b1 = svenns_algorithm(point, S)
        if dsc==1:
            LAMBDA = dsc_powell(point, S, a1, b1)
        else:
            LAMBDA = golden_ratio(point, S, a1, b1)
        point = add(prev_x_3, mult(LAMBDA, S))
        print(f'x^({i+1}) = {point}^T')
        if norm(grad_pr(point)) < epsilon:
            return None
        point_history.append(point)
        prev_x_4 = point 
        prev_x_0, prev_x_2, prev_x_1 = prev_x_1, prev_x_4, prev_x_3
        i+=1
        if i >= lim:
            break
    x_vals = [point[0] for point in point_history]
    y_vals = [point[1] for point in point_history]

    x = np.linspace(min(x_vals) - 1, max(x_vals) + 1, 100)
    y = np.linspace(min(y_vals) - 1, max(y_vals) + 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = f((X, Y))

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=50)
    plt.plot(x_vals, y_vals, marker='o', color='r', label='Optimization Path')
    plt.colorbar()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Contour Plot with Optimization Path')
    plt.legend()
    plt.grid(True)
    plt.show()
    return None

def partan_mod_c_norm(point,epsilon=0.00001,lim=20, dsc=1):
    i=0
    point_history = []
    point_history.append(point)
    prev_x_0 = point
    S = mult(-1, grad_c(point))
    a1,b1 = svenns_algorithm(point, S)
    if dsc==1:
        LAMBDA = dsc_powell(point, S, a1, b1)
    else:
        LAMBDA = golden_ratio(point, S, a1, b1)
    point = add(prev_x_0, mult(LAMBDA, S))
    print(f'x^({i+1}) = {point}^T')
    if norm(grad_c(point)) < epsilon:
        return None
    point_history.append(point)
    prev_x_1 = point
    S_1 = mult(-1, grad_c(point))
    a2,b2 = svenns_algorithm(point, S_1)
    if dsc==1:
        LAMBDA = dsc_powell(point, S_1, a2, b2)
    else:
        LAMBDA = golden_ratio(point, S_1, a2, b2)
    point = add(prev_x_1, mult(LAMBDA, S_1))
    print(f'x^({i+2}) = {point}^T')
    if norm(grad_c(point)) < epsilon:
        return None 
    point_history.append(point)
    prev_x_2 = point
    i += 2
    while norm(grad_c(prev_x_2)) > epsilon:
        S_2 = add(prev_x_2, mult(-1, prev_x_0))
        a3,b3 = svenns_algorithm(point, S_2)
        if dsc==1:
            LAMBDA = dsc_powell(point, S_2, a3, b3)
        else:
            LAMBDA = golden_ratio(point, S_2, a3, b3)
        point = add(prev_x_2, mult(LAMBDA, S_2))
        point_history.append(point)
        prev_x_3 = point
        print(f'x^({i+1}) = {point}^T')
        if norm(grad_c(point)) < epsilon:
            return None
        i += 1
        if norm(grad_c(point)) <= epsilon:
            break
        S = mult(-1, grad_c(point))
        a1,b1 = svenns_algorithm(point, S)
        if dsc==1:
            LAMBDA = dsc_powell(point, S, a1, b1)
        else:
            LAMBDA = golden_ratio(point, S, a1, b1)
        point = add(prev_x_3, mult(LAMBDA, S))
        print(f'x^({i+1}) = {point}^T')
        if norm(grad_c(point)) < epsilon:
            return None
        point_history.append(point)
        prev_x_4 = point 
        prev_x_0, prev_x_2, prev_x_1 = prev_x_1, prev_x_4, prev_x_3
        i+=1
        if i >= lim:
            break
    x_vals = [point[0] for point in point_history]
    y_vals = [point[1] for point in point_history]

    x = np.linspace(min(x_vals) - 1, max(x_vals) + 1, 100)
    y = np.linspace(min(y_vals) - 1, max(y_vals) + 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = f((X, Y))

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=50)
    plt.plot(x_vals, y_vals, marker='o', color='r', label='Optimization Path')
    plt.colorbar()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Contour Plot with Optimization Path')
    plt.legend()
    plt.grid(True)
    plt.show()
    return None

def partan_mod_l_norm(point,epsilon=0.00001,lim=20, dsc=1):
    i=0
    point_history = []
    point_history.append(point)
    prev_x_0 = point
    S = mult(-1, grad_l(point))
    a1,b1 = svenns_algorithm(point, S)
    if dsc==1:
        LAMBDA = dsc_powell(point, S, a1, b1)
    else:
        LAMBDA = golden_ratio(point, S, a1, b1)
    point = add(prev_x_0, mult(LAMBDA, S))
    print(f'x^({i+1}) = {point}^T')
    if norm(grad_l(point)) < epsilon:
        return None
    point_history.append(point)
    prev_x_1 = point
    S_1 = mult(-1, grad_l(point))
    a2,b2 = svenns_algorithm(point, S_1)
    if dsc==1:
        LAMBDA = dsc_powell(point, S_1, a2, b2)
    else:
        LAMBDA = golden_ratio(point, S_1, a2, b2)
    point = add(prev_x_1, mult(LAMBDA, S_1))
    print(f'x^({i+2}) = {point}^T')
    if norm(grad_l(point)) < epsilon:
        return None
    point_history.append(point)
    prev_x_2 = point
    i += 2
    while norm(grad_l(prev_x_2)) > epsilon:
        S_2 = add(prev_x_2, mult(-1, prev_x_0))
        a3,b3 = svenns_algorithm(point, S_2)
        if dsc==1:
            LAMBDA = dsc_powell(point, S_2, a3, b3)
        else:
            LAMBDA = golden_ratio(point, S_2, a3, b3)
        point = add(prev_x_2, mult(LAMBDA, S_2))
        point_history.append(point)
        prev_x_3 = point
        print(f'x^({i+1}) = {point}^T')
        if norm(grad_l(point)) < epsilon:
            return None
        i += 1
        if norm(grad_l(point)) <= epsilon:
            break
        S = mult(-1, grad_l(point))
        a1,b1 = svenns_algorithm(point, S)
        if dsc==1:
            LAMBDA = dsc_powell(point, S, a1, b1)
        else:
            LAMBDA = golden_ratio(point, S, a1, b1)
        point = add(prev_x_3, mult(LAMBDA, S))
        print(f'x^({i+1}) = {point}^T')
        if norm(grad_l(point)) < epsilon:
            return None
        point_history.append(point)
        prev_x_4 = point 
        prev_x_0, prev_x_2, prev_x_1 = prev_x_1, prev_x_4, prev_x_3
        i+=1
        if i >= lim:
            break
    x_vals = [point[0] for point in point_history]
    y_vals = [point[1] for point in point_history]

    x = np.linspace(min(x_vals) - 1, max(x_vals) + 1, 100)
    y = np.linspace(min(y_vals) - 1, max(y_vals) + 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = f((X, Y))

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=50)
    plt.plot(x_vals, y_vals, marker='o', color='r', label='Optimization Path')
    plt.colorbar()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Contour Plot with Optimization Path')
    plt.legend()
    plt.grid(True)
    plt.show()
    return None

def mns_pr_other(point, lim=20, epsilon=0.01, dsc=1):
    i=0
    point_history = []
    prev_x = point
    while (norm(add(point, mult(-1,prev_x))))/(norm(prev_x)) > epsilon or abs(f(point) - f(prev_x)) > epsilon or i < lim:
        point_history.append(point)
        prev_x = point
        S = mult(-1, grad_pr(point))
        a1,b1 = svenns_algorithm(point, S)
        if dsc==1:
            LAMBDA = dsc_powell(point, S, a1, b1)
        else:
            LAMBDA = golden_ratio(point, S, a1, b1)
        point = add(prev_x, mult(LAMBDA, S))
        print(f'x^({i+1}) = {point}^T')
        if (norm(add(point, mult(-1,prev_x))))/(norm(prev_x)) < epsilon and abs(f(point) - f(prev_x)) < epsilon:
            break 
        point_history.append(point)
        if i >= lim - 1:
            break
        i += 1
    x_vals = [point[0] for point in point_history]
    y_vals = [point[1] for point in point_history]

    x = np.linspace(min(x_vals) - 1, max(x_vals) + 1, 100)
    y = np.linspace(min(y_vals) - 1, max(y_vals) + 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = f((X, Y))

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=50)
    plt.plot(x_vals, y_vals, marker='o', color='r', label='Optimization Path')
    plt.colorbar()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Contour Plot with Optimization Path')
    plt.legend()
    plt.grid(True)
    plt.show()
    return None

def mns_c_other(point, lim=20, epsilon=0.01, dsc=1):
    i=0
    point_history = []
    prev_x = point
    while (norm(add(point, mult(-1,prev_x))))/(norm(prev_x)) > epsilon or abs(f(point) - f(prev_x)) > epsilon or i < lim:
        point_history.append(point)
        prev_x = point
        S = mult(-1, grad_c(point))
        a1,b1 = svenns_algorithm(point, S)
        if dsc==1:
            LAMBDA = dsc_powell(point, S, a1, b1)
        else:
            LAMBDA = golden_ratio(point, S, a1, b1)
        point = add(prev_x, mult(LAMBDA, S))
        print(f'x^({i+1}) = {point}^T')
        if (norm(add(point, mult(-1,prev_x))))/(norm(prev_x)) < epsilon and abs(f(point) - f(prev_x)) < epsilon:
            break 
        point_history.append(point)
        if i >= lim - 1:
            break
        i += 1
    x_vals = [point[0] for point in point_history]
    y_vals = [point[1] for point in point_history]

    x = np.linspace(min(x_vals) - 1, max(x_vals) + 1, 100)
    y = np.linspace(min(y_vals) - 1, max(y_vals) + 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = f((X, Y))

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=50)
    plt.plot(x_vals, y_vals, marker='o', color='r', label='Optimization Path')
    plt.colorbar()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Contour Plot with Optimization Path')
    plt.legend()
    plt.grid(True)
    plt.show()
    return None

def mns_l_other(point, lim=20, epsilon=0.01, dsc=1):
    i=0
    point_history = []
    prev_x = point
    while (norm(add(point, mult(-1,prev_x))))/(norm(prev_x)) > epsilon or abs(f(point) - f(prev_x)) > epsilon or i < lim:
        point_history.append(point)
        prev_x = point
        S = mult(-1, grad_l(point))
        a1,b1 = svenns_algorithm(point, S)
        if dsc==1:
            LAMBDA = dsc_powell(point, S, a1, b1)
        else:
            LAMBDA = golden_ratio(point, S, a1, b1)
        point = add(prev_x, mult(LAMBDA, S))
        print(f'x^({i+1}) = {point}^T')
        if (norm(add(point, mult(-1,prev_x))))/(norm(prev_x)) < epsilon and abs(f(point) - f(prev_x)) < epsilon:
            break 
        point_history.append(point)
        if i >= lim - 1:
            break
        i += 1
    x_vals = [point[0] for point in point_history]
    y_vals = [point[1] for point in point_history]

    x = np.linspace(min(x_vals) - 1, max(x_vals) + 1, 100)
    y = np.linspace(min(y_vals) - 1, max(y_vals) + 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = f((X, Y))

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=50)
    plt.plot(x_vals, y_vals, marker='o', color='r', label='Optimization Path')
    plt.colorbar()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Contour Plot with Optimization Path')
    plt.legend()
    plt.grid(True)
    plt.show()
    return None

def partan_c_other(point,epsilon=0.001,lim=20, dsc=1):
    i=0
    point_history = []
    prev_x = point
    while (norm(add(point, mult(-1,prev_x))))/(norm(prev_x)) > epsilon or abs(f(point) - f(prev_x)) > epsilon or i < lim:
        point_history.append(point)
        prev_x = point
        S = mult(-1, grad_c(point))
        a1,b1 = svenns_algorithm(point, S, lambda0=0)
        if dsc==1:
            LAMBDA = dsc_powell(point, S, a1, b1, e=0.0001)
        else:
            LAMBDA = golden_ratio(point, S, a1, b1)
        point = add(prev_x, mult(LAMBDA, S))
        print(f'x^({i+1}) = {point}^T')
        if (norm(add(point, mult(-1,prev_x))))/(norm(prev_x)) < epsilon and abs(f(point) - f(prev_x)) < epsilon:
            break 
        point_history.append(point)
        prev_x_1 = point
        S_1 = mult(-1, grad_c(point))
        a2,b2 = svenns_algorithm(point, S_1,lambda0=0)
        if dsc==1:
            LAMBDA = dsc_powell(point, S_1, a2, b2, e=0.0001)
        else:
            LAMBDA = golden_ratio(point, S_1, a2, b2)
        point = add(prev_x_1, mult(LAMBDA, S_1))
        print(f'x^({i+2}) = {point}^T')
        if (norm(add(point, mult(-1,prev_x_1))))/(norm(prev_x_1)) < epsilon and abs(f(point) - f(prev_x_1)) < epsilon:
            break
        point_history.append(point)
        prev_x_2 = point
        S_2 = add(prev_x_1, mult(-1,prev_x))
        a3,b3 = svenns_algorithm(point, S_2)
        if dsc==1:
            LAMBDA = dsc_powell(point, S_2, a3, b3, e=0.0001)
        else:
            LAMBDA = golden_ratio(point, S_2, a3, b3)
        point = add(prev_x_2, mult(LAMBDA, S_2))
        print(f'x^({i+3}) = {point}^T')
        if (norm(add(point, mult(-1,prev_x_2))))/(norm(prev_x_2)) < epsilon and abs(f(point) - f(prev_x_2)) < epsilon:
            break
        point_history.append(point)
        if i >= lim:
            break
        i += 3
    x_vals = [point[0] for point in point_history]
    y_vals = [point[1] for point in point_history]

    x = np.linspace(min(x_vals) - 1, max(x_vals) + 1, 100)
    y = np.linspace(min(y_vals) - 1, max(y_vals) + 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = f((X, Y))

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=50)
    plt.plot(x_vals, y_vals, marker='o', color='r', label='Optimization Path')
    plt.colorbar()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Contour Plot with Optimization Path')
    plt.legend()
    plt.grid(True)
    plt.show()

def partan_pr_other(point,epsilon=0.001,lim=20, dsc=1):
    i=0
    point_history = []
    prev_x = point
    while (norm(add(point, mult(-1,prev_x))))/(norm(prev_x)) > epsilon or abs(f(point) - f(prev_x)) > epsilon or i < lim:
        point_history.append(point)
        prev_x = point
        S = mult(-1, grad_pr(point))
        a1,b1 = svenns_algorithm(point, S)
        if dsc==1:
            LAMBDA = dsc_powell(point, S, a1, b1)
        else:
            LAMBDA = golden_ratio(point, S, a1, b1)
        point = add(prev_x, mult(LAMBDA, S))
        print(f'x^({i+1}) = {point}^T')
        if (norm(add(point, mult(-1,prev_x))))/(norm(prev_x)) < epsilon and abs(f(point) - f(prev_x)) < epsilon:
            break
        point_history.append(point)
        prev_x_1 = point
        S_1 = mult(-1, grad_pr(point))
        a2,b2 = svenns_algorithm(point, S_1)
        if dsc==1:
            LAMBDA = dsc_powell(point, S_1, a2, b2)
        else:
            LAMBDA = golden_ratio(point, S_1, a2, b2)
        point = add(prev_x_1, mult(LAMBDA, S_1))
        print(f'x^({i+2}) = {point}^T')
        if (norm(add(point, mult(-1,prev_x_1))))/(norm(prev_x_1)) < epsilon and abs(f(point) - f(prev_x_1)) < epsilon:
            break
        point_history.append(point)
        prev_x_2 = point
        S_2 = add(prev_x_1, mult(-1,prev_x))
        a3,b3 = svenns_algorithm(point, S_2)
        if dsc==1:
            LAMBDA = dsc_powell(point, S_2, a3, b3)
        else:
            LAMBDA = golden_ratio(point, S_2, a3, b3)
        point = add(prev_x_2, mult(LAMBDA, S_2))
        print(f'x^({i+3}) = {point}^T')
        if (norm(add(point, mult(-1,prev_x_2))))/(norm(prev_x_2)) < epsilon and abs(f(point) - f(prev_x_2)) < epsilon:
            break
        point_history.append(point)
        if i >= lim:
            break
        i += 3
    x_vals = [point[0] for point in point_history]
    y_vals = [point[1] for point in point_history]

    x = np.linspace(min(x_vals) - 1, max(x_vals) + 1, 100)
    y = np.linspace(min(y_vals) - 1, max(y_vals) + 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = f((X, Y))

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=50)
    plt.plot(x_vals, y_vals, marker='o', color='r', label='Optimization Path')
    plt.colorbar()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Contour Plot with Optimization Path')
    plt.legend()
    plt.grid(True)
    plt.show()

def partan_l_other(point,epsilon=0.001,lim=20, dsc=1):
    i=0
    point_history = []
    prev_x = point
    while (norm(add(point, mult(-1,prev_x))))/(norm(prev_x)) > epsilon or abs(f(point) - f(prev_x)) > epsilon or i < lim:
        point_history.append(point)
        prev_x = point
        S = mult(-1, grad_l(point))
        a1,b1 = svenns_algorithm(point, S, lambda0=0)
        if dsc==1:
            LAMBDA = dsc_powell(point, S, a1, b1, e=0.0001)
        else:
            LAMBDA = golden_ratio(point, S, a1, b1)
        point = add(prev_x, mult(LAMBDA, S))
        print(f'x^({i+1}) = {point}^T')
        if (norm(add(point, mult(-1,prev_x))))/(norm(prev_x)) < epsilon and abs(f(point) - f(prev_x)) < epsilon:
            break 
        point_history.append(point)
        prev_x_1 = point
        S_1 = mult(-1, grad_l(point))
        a2,b2 = svenns_algorithm(point, S_1,lambda0=0)
        if dsc==1:
            LAMBDA = dsc_powell(point, S_1, a2, b2, e=0.0001)
        else:
            LAMBDA = golden_ratio(point, S_1, a2, b2)
        point = add(prev_x_1, mult(LAMBDA, S_1))
        print(f'x^({i+2}) = {point}^T')
        if (norm(add(point, mult(-1,prev_x_1))))/(norm(prev_x_1)) < epsilon and abs(f(point) - f(prev_x_1)) < epsilon:
            break 
        point_history.append(point)
        prev_x_2 = point
        S_2 = add(prev_x_1, mult(-1,prev_x))
        a3,b3 = svenns_algorithm(point, S_2)
        if dsc==1:
            LAMBDA = dsc_powell(point, S_2, a3, b3, e=0.0001)
        else:
            LAMBDA = golden_ratio(point, S_2, a3, b3)
        point = add(prev_x_2, mult(LAMBDA, S_2))
        print(f'x^({i+3}) = {point}^T')
        if (norm(add(point, mult(-1,prev_x_2))))/(norm(prev_x_2)) < epsilon and abs(f(point) - f(prev_x_2)) < epsilon:
            break
        point_history.append(point)
        if i >= lim:
            break
        i += 3
    x_vals = [point[0] for point in point_history]
    y_vals = [point[1] for point in point_history]

    x = np.linspace(min(x_vals) - 1, max(x_vals) + 1, 100)
    y = np.linspace(min(y_vals) - 1, max(y_vals) + 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = f((X, Y))

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=50)
    plt.plot(x_vals, y_vals, marker='o', color='r', label='Optimization Path')
    plt.colorbar()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Contour Plot with Optimization Path')
    plt.legend()
    plt.grid(True)
    plt.show()

def partan_mod_c_other(point,epsilon=0.0001,lim=20, dsc=1):
    i=0
    point_history = []
    point_history.append(point)
    prev_x_0 = point
    S = mult(-1, grad_c(point))
    a1,b1 = svenns_algorithm(point, S)
    if dsc==1:
        LAMBDA = dsc_powell(point, S, a1, b1)
    else:
        LAMBDA = golden_ratio(point, S, a1, b1)
    point = add(prev_x_0, mult(LAMBDA, S))
    print(f'x^({i+1}) = {point}^T')
    point_history.append(point)
    prev_x_1 = point
    S_1 = mult(-1, grad_c(point))
    a2,b2 = svenns_algorithm(point, S_1)
    if dsc==1:
        LAMBDA = dsc_powell(point, S_1, a2, b2)
    else:
        LAMBDA = golden_ratio(point, S_1, a2, b2)
    point = add(prev_x_1, mult(LAMBDA, S_1))
    print(f'x^({i+2}) = {point}^T')
    if (norm(add(point, mult(-1,prev_x_1))))/(norm(prev_x_1)) < epsilon and abs(f(point) - f(prev_x_1)) < epsilon:
        return None
    point_history.append(point) 
    point_history.append(point)
    prev_x_2 = point
    i += 2
    while (norm(add(point, mult(-1,prev_x_2))))/(norm(prev_x_2)) > epsilon or abs(f(point) - f(prev_x_2)) > epsilon or i < lim:
        S_2 = add(prev_x_2, mult(-1, prev_x_0))
        a3,b3 = svenns_algorithm(point, S_2)
        if dsc==1:
            LAMBDA = dsc_powell(point, S_2, a3, b3)
        else:
            LAMBDA = golden_ratio(point, S_2, a3, b3)
        point = add(prev_x_2, mult(LAMBDA, S_2))
        point_history.append(point)
        prev_x_3 = point
        print(f'x^({i+1}) = {point}^T')
        if (norm(add(point, mult(-1,prev_x_2))))/(norm(prev_x_2)) < epsilon and abs(f(point) - f(prev_x_2)) < epsilon:
            break       
        i += 1
        if norm(grad_c(point)) <= epsilon:
            break
        S = mult(-1, grad_c(point))
        a1,b1 = svenns_algorithm(point, S)
        if dsc==1:
            LAMBDA = dsc_powell(point, S, a1, b1)
        else:
            LAMBDA = golden_ratio(point, S, a1, b1)
        point = add(prev_x_3, mult(LAMBDA, S))
        print(f'x^({i+1}) = {point}^T')
        if (norm(add(point, mult(-1,prev_x_3))))/(norm(prev_x_3)) < epsilon and abs(f(point) - f(prev_x_3)) < epsilon:
            break
        point_history.append(point)
        prev_x_4 = point 
        prev_x_0, prev_x_2, prev_x_1 = prev_x_1, prev_x_4, prev_x_3
        i+=1
        if i >= lim:
            break
    x_vals = [point[0] for point in point_history]
    y_vals = [point[1] for point in point_history]

    x = np.linspace(min(x_vals) - 1, max(x_vals) + 1, 100)
    y = np.linspace(min(y_vals) - 1, max(y_vals) + 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = f((X, Y))

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=50)
    plt.plot(x_vals, y_vals, marker='o', color='r', label='Optimization Path')
    plt.colorbar()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Contour Plot with Optimization Path')
    plt.legend()
    plt.grid(True)
    plt.show()
    return None


def partan_mod_pr_other(point,epsilon=0.0001,lim=20, dsc=1):
    i=0
    point_history = []
    point_history.append(point)
    prev_x_0 = point
    S = mult(-1, grad_pr(point))
    a1,b1 = svenns_algorithm(point, S)
    if dsc==1:
        LAMBDA = dsc_powell(point, S, a1, b1)
    else:
        LAMBDA = golden_ratio(point, S, a1, b1)
    point = add(prev_x_0, mult(LAMBDA, S))
    print(f'x^({i+1}) = {point}^T')
    point_history.append(point)
    prev_x_1 = point
    S_1 = mult(-1, grad_pr(point))
    a2,b2 = svenns_algorithm(point, S_1)
    if dsc==1:
        LAMBDA = dsc_powell(point, S_1, a2, b2)
    else:
        LAMBDA = golden_ratio(point, S_1, a2, b2)
    point = add(prev_x_1, mult(LAMBDA, S_1))
    print(f'x^({i+2}) = {point}^T')
    if (norm(add(point, mult(-1,prev_x_1))))/(norm(prev_x_1)) < epsilon and abs(f(point) - f(prev_x_1)) < epsilon:
        return None
    point_history.append(point) 
    point_history.append(point)
    prev_x_2 = point
    i += 2
    while (norm(add(point, mult(-1,prev_x_2))))/(norm(prev_x_2)) > epsilon or abs(f(point) - f(prev_x_2)) > epsilon or i < lim:
        S_2 = add(prev_x_2, mult(-1, prev_x_0))
        a3,b3 = svenns_algorithm(point, S_2)
        if dsc==1:
            LAMBDA = dsc_powell(point, S_2, a3, b3)
        else:
            LAMBDA = golden_ratio(point, S_2, a3, b3)
        point = add(prev_x_2, mult(LAMBDA, S_2))
        point_history.append(point)
        prev_x_3 = point
        print(f'x^({i+1}) = {point}^T')
        if (norm(add(point, mult(-1,prev_x_2))))/(norm(prev_x_2)) < epsilon and abs(f(point) - f(prev_x_2)) < epsilon:
            break       
        i += 1
        if norm(grad_pr(point)) <= epsilon:
            break
        S = mult(-1, grad_pr(point))
        a1,b1 = svenns_algorithm(point, S)
        if dsc==1:
            LAMBDA = dsc_powell(point, S, a1, b1)
        else:
            LAMBDA = golden_ratio(point, S, a1, b1)
        point = add(prev_x_3, mult(LAMBDA, S))
        print(f'x^({i+1}) = {point}^T')
        if (norm(add(point, mult(-1,prev_x_3))))/(norm(prev_x_3)) < epsilon and abs(f(point) - f(prev_x_3)) < epsilon:
            break
        point_history.append(point)
        prev_x_4 = point 
        prev_x_0, prev_x_2, prev_x_1 = prev_x_1, prev_x_4, prev_x_3
        i+=1
        if i >= lim:
            break
    x_vals = [point[0] for point in point_history]
    y_vals = [point[1] for point in point_history]

    x = np.linspace(min(x_vals) - 1, max(x_vals) + 1, 100)
    y = np.linspace(min(y_vals) - 1, max(y_vals) + 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = f((X, Y))

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=50)
    plt.plot(x_vals, y_vals, marker='o', color='r', label='Optimization Path')
    plt.colorbar()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Contour Plot with Optimization Path')
    plt.legend()
    plt.grid(True)
    plt.show()
    return None

def partan_mod_l_other(point,epsilon=0.0001,lim=20, dsc=1):
    i=0
    point_history = []
    point_history.append(point)
    prev_x_0 = point
    S = mult(-1, grad_l(point))
    a1,b1 = svenns_algorithm(point, S)
    if dsc==1:
        LAMBDA = dsc_powell(point, S, a1, b1)
    else:
        LAMBDA = golden_ratio(point, S, a1, b1)
    point = add(prev_x_0, mult(LAMBDA, S))
    print(f'x^({i+1}) = {point}^T')
    point_history.append(point)
    prev_x_1 = point
    S_1 = mult(-1, grad_l(point))
    a2,b2 = svenns_algorithm(point, S_1)
    if dsc==1:
        LAMBDA = dsc_powell(point, S_1, a2, b2)
    else:
        LAMBDA = golden_ratio(point, S_1, a2, b2)
    point = add(prev_x_1, mult(LAMBDA, S_1))
    print(f'x^({i+2}) = {point}^T')
    if (norm(add(point, mult(-1,prev_x_1))))/(norm(prev_x_1)) < epsilon and abs(f(point) - f(prev_x_1)) < epsilon:
        return None
    point_history.append(point) 
    point_history.append(point)
    prev_x_2 = point
    i += 2
    while (norm(add(point, mult(-1,prev_x_2))))/(norm(prev_x_2)) > epsilon or abs(f(point) - f(prev_x_2)) > epsilon or i < lim:
        S_2 = add(prev_x_2, mult(-1, prev_x_0))
        a3,b3 = svenns_algorithm(point, S_2)
        if dsc==1:
            LAMBDA = dsc_powell(point, S_2, a3, b3)
        else:
            LAMBDA = golden_ratio(point, S_2, a3, b3)
        point = add(prev_x_2, mult(LAMBDA, S_2))
        point_history.append(point)
        prev_x_3 = point
        print(f'x^({i+1}) = {point}^T')
        if (norm(add(point, mult(-1,prev_x_2))))/(norm(prev_x_2)) < epsilon and abs(f(point) - f(prev_x_2)) < epsilon:
            break       
        i += 1
        if norm(grad_l(point)) <= epsilon:
            break
        S = mult(-1, grad_l(point))
        a1,b1 = svenns_algorithm(point, S)
        if dsc==1:
            LAMBDA = dsc_powell(point, S, a1, b1)
        else:
            LAMBDA = golden_ratio(point, S, a1, b1)
        point = add(prev_x_3, mult(LAMBDA, S))
        print(f'x^({i+1}) = {point}^T')
        if (norm(add(point, mult(-1,prev_x_3))))/(norm(prev_x_3)) < epsilon and abs(f(point) - f(prev_x_3)) < epsilon:
            break
        point_history.append(point)
        prev_x_4 = point 
        prev_x_0, prev_x_2, prev_x_1 = prev_x_1, prev_x_4, prev_x_3
        i+=1
        if i >= lim:
            break
    x_vals = [point[0] for point in point_history]
    y_vals = [point[1] for point in point_history]

    x = np.linspace(min(x_vals) - 1, max(x_vals) + 1, 100)
    y = np.linspace(min(y_vals) - 1, max(y_vals) + 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = f((X, Y))

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=50)
    plt.plot(x_vals, y_vals, marker='o', color='r', label='Optimization Path')
    plt.colorbar()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Contour Plot with Optimization Path')
    plt.legend()
    plt.grid(True)
    plt.show()
    return None