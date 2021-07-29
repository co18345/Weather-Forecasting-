from __future__ import division
import csv, sys, re, timeit, math
from sklearn import datasets, linear_model, preprocessing, neural_network
from sklearn.utils import column_or_1d
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import os
import errno
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from matplotlib import dates as mPlotDATEs
    try: 
        int(s)
        return True
    except ValueError:
        return False

def RepresentsFloat(s):
    try: 
        float(s)
        return True
    except ValueError:
        return False

elif (option == 'hour'):
        time_diff = time_obj - data time (time_obj.year, time_obj.month, time_obj.day, time_obj.hour, 0)
    other:
        expand SystemError ("option is not a valid thread!")
        
    return int (time_diff.total_seconds () / 60)

# do the translation
def interpolate_df (df, features):
    df_re = df
    
    print ("len (df.index) = {}". format (len (df.index)))
    
    # check all data float data and convert the data type to float64
    for col in features:
        # df [col] = df [col] .astype (float)
        temp = df [df [col] .isnull ()]
        #print (test title)
        Print ("===")
        # print (test title (n = 1))
        print ("{} type {}". format (col, df [col] .dtype))
        print ("{} type contains {} np.NaN". format (col, len (temp.index)))
        Print ("===")
    
    
    
    print ("len (df.index) = {}". format (len (df.index)))
    # can be time to use as a reference and set method = 'time'
    # df.to_csv ("df_before_interpolate.csv")
    # df [features] = df [features] .interpolate (method = 'time')
    # df.loc [:, features] = df [features] .interpolate (method = 'time')
    # somehow, df (input) will be updated or used inplace = False
    df_re.loc [:, features] = df [features] .interpolate (method = 'time', inplace = False)
    # df.to_csv ("df_after_interpolate.csv")
    #print ("df =")
    #print (df)
    
    # capture real nan values
    df_nan_interpolate = df.loc [df_nan.index.values]
    print ("len (df_nan_interpolate.index) = {}". format (len (df_nan_interpolate.index)))
    df_nan_interpolate.to_csv ("df_nan_interpolate.csv")
    
    if (df_re.notnull (). all (axis = 1) .all (axis = 0)):
        print ("CHECK: No value in df_re.")
        
    replace df_re

    df_train = df.loc [(df.index> data_start) & (df.index <= data_end) ,:]]
    
    # do interpolate on training set only
    df_train = interpolate_df (df_train, features)
    df_train.to_csv ('df_train_clean.csv')
    
    X_train = df_train [features]
    y_train = df_train [target]
    
    # configure test data
    data_start = time (data_test_yr_start, 1, 1, 0, 0, 0)
    data_end = time (data_test_yr_end, 12, 31, 23, 59, 59)
    df_test = df.loc [(df.index> data_start) & (df.index <= data_end) ,:]]
    
    # drops the number lines of the NaN test set
    (irowu_old, col_old) = df_test.shape
    print ("Before you draw the NaN number of the test set, df_test.shape = {}". format (df_test.shape))
    df_test = df_test [df_test.notnull (). all (axis = 1)]
    (row, column) = df_test.shape
    print ("After dragging the NaN number of the test set, df_test.shape = {}". format (df_test.shape))
    print ("Drop Level = {0: .2f}". format (float (1 - (line / line_knee))))
    
    df_test.to_csv ('df_test_clean.csv')
    X_test = df_test [features]
    y_test = df_test [stones]
    
    # familiarity and training scale / test set
    # use robust_scaler to protect misleading merchants
    # scaler = processing StandardScaler ()
    # use robust_scaler to protect misleading merchants
    scale = advancement.RebustScaler ()
    X_train = scaler.fit_transform (X_train)
    X_test = scaler.transform (X_test)
    
    return (X_train, y_train, X_test, y_test)
    
def normalization (df_train, df_test, targets, features):
    
    # use robust_scaler to protect misleading merchants
    scale = advancement.RebustScaler ()
    X_train = scaler.fit_transform (X_train)
    X_test = scaler.transform (X_test)
    
    return (X_train, y_train, X_test, y_test)

# structure y_test
def plot_y_test (regr, X_test, y_test, ask_user):
    (r_test, c_test) = X_test.shape
    
    # for i in range (c_test):
    # plt. scatter (X_test [:, i], y_test)
    # plt.plot (X_test [:, i], regr.predict (X_test), color = 'blue', linewidth = 3)
    
    y_predict = regr.predict (X_test)
    #print ("==> y_test type = {}". format (type (y_test)))
    #print ("y_test.index = {}". format (y_test.index))
    #print ("y_test = {}". format (y_test))
    #print ("y_predict = {}". format (y_predict))
    df_plot = y_test
    #print (df_plot)
    #print ("DATE")
    #print ("# # # # # # # # # # # # # # # # # # #
    df_plot = df_plot.reset_index (level = ['DATE'])
    df_plot.loc [:, 'predict_temp_C'] = y_cognize
    # go back green DATE time 1day_later
    df_plot.loc [:, "raw_DATE"] = df_plot ['DATE']. apply (lambda time_obj: time_obj + relativedelta (days = 1))
    df_plot.rename (columns = {'1days_later_temp_C': 'raw_temp_C', 'DATE': 'label_DATE'}, inplace = True)
    
    df_plot = df_plot.set_index ("green_DATE")
    #print (df_plot)
    #print ("# # # # # # # # # # # # # # # # # #
    
    # the time range for the default layout
    icebo_yr = 2016
    structure_month = 10
    icebo_day = 5
    duration = 10
    
    df_test.to_csv('df_test_clean.csv')
    X_test = df_test[features] 
    y_test = df_test[targets] 
    
    # normalization and scale for training/test set
    # use robust_scaler to avoid misleading outliers
    # scaler = preprocessing.StandardScaler()
    # use robust_scaler to avoid misleading outliers
    scaler = preprocessing.RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return (X_train, y_train, X_test, y_test)
    
def normalization(df_train, df_test, targets, features):
    
    # use robust_scaler to avoid misleading outliers
    scaler = preprocessing.RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return (X_train, y_train, X_test, y_test)

# plot y_test
def plot_y_test(regr, X_test, y_test, ask_user):
    (r_test, c_test) = X_test.shape
    
    # for i in range(c_test):
    #     plt.scatter(X_test[:, i], y_test)
    #     plt.plot(X_test[:, i], regr.predict(X_test), color='blue', linewidth=3)
    
    y_predict = regr.predict(X_test)
    # print("==> y_test type = {}".format(type(y_test)) )
    # print("y_test.index = {}".format(y_test.index))
    # print("y_test = {}".format(y_test) )
    # print("y_predict = {}".format(y_predict) )
    df_plot = y_test
    # print(df_plot)
    # print("DATE")
    # print("#########################")
    df_plot = df_plot.reset_index(level=['DATE'])
    df_plot.loc[:,'predict_temp_C'] = y_predict
    # shift back to raw DATE time 1day_later
    df_plot.loc[:,"raw_DATE"] = df_plot['DATE'].apply(lambda time_obj: time_obj + relativedelta(days=1))
    df_plot.rename(columns={'1days_later_temp_C': 'raw_temp_C', 'DATE':'label_DATE'}, inplace=True)
    
    df_plot = df_plot.set_index("raw_DATE")
    # print(df_plot)
    # print("#########################")
    
    # default plot time range
    plot_yr = 2016
    plot_month = 10
    plot_day = 5 
    duration = 10 
    
    range_start = datetime(plot_yr, plot_month, plot_day, 0, 0, 0)
    range_end = datetime(plot_yr, plot_month, plot_day, 0, 0, 0) + relativedelta(days=duration)
    
    if (range_start < datetime(2016,1,2,0,0,0) or range_end > datetime(2017,1,1,0,0,0) ):
        raise SystemExit("Input date is out of range! Please try again!")
    else:
        print("Correct format and time range!")
        
    if (ask_user == True):
)
        print("years, month, day, ploting duration(days) \n")
        print("For example, enter: {}, {}, {}, {}".format(plot_yr, plot_month, plot_day, duration) )
        
        input_format_ok = False
        while(input_format_ok == False):
            user_input = input()
            print("Your input is {}".format(user_input) )
            try:
                plot_yr = int(user_input[0]) 
                plot_month = int(user_input[1]) 
                plot_day = int(user_input[2]) 
                duration = int(user_input[3]) 
                
                range_start = datetime(plot_yr, plot_month, plot_day, 0, 0, 0)
                range_end = datetime(plot_yr, plot_month, plot_day, 0, 0, 0) + relativedelta(days=duration)
                
                if (range_start < datetime(2016,1,2,0,0,0) or range_end > datetime(2017,1,1,0,0,0) ):
                    print("Input date is out of range! Please try again!")
                else:
                    print("Correct format and time range!")
                    input_format_ok = True
            except:
                print("Incorrect format, please try again!")
    
    df_plot = df_plot[range_start.strftime('%Y-%m-%d %H:%M:%S') : range_end.strftime('%Y-%m-%d %H:%M:%S')] 
DateFormatter('%Y-%m-%d %H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)
    ax.xaxis_date()
    # plt.scatter(y_test.index, y_test)
    # plt.plot(y_test.index, y_predict, color='blue', linewidth=3)
    # plt.scatter(y_test.index[0:25], y_test[0:25])
    # plt.plot(y_test.index[0:25], y_test[0:25], color='red', linewidth=3)
    # plt.plot(y_test.index[0:25], y_predict[0:25], color='blue', linewidth=3)
    # plt.subplot(121)
    plt.xlabel("time range")
    plt.ylabel("degree C")
    plt.title("raw data (red) v.s. predict data (blue)") 
    plt.grid()
    plt.plot(datenums, value_raw, linestyle='-', marker='o', markersize=5, color='r', linewidth=2, label="raw temp C")
    plt.plot(datenums, value_predict, linestyle='-', marker='o', markersize=5, color='b', linewidth=2, label="predict temp C")
    plt.legend(loc="best")
    
    plt.show()
    
    plt.figure()
    # plt.subplot(122)
    plt.xlabel("raw data (degree C)")
    plt.ylabel("predict data (degree C)")
    plt.title("perfect match (red) v.s. model (blue)") 
    plt.grid()
    plt.plot(value_raw, value_raw, linestyle='--', marker='o', markersize=5, color='r', linewidth=1, label="perfect match line")
    plt.scatter(value_raw, value_predict, marker='o', s=10, color='b', label="predict temp C")
    # plt.plot(value_predict, marker='o', markersize=3, color='b', label="predict temp C")
    plt.legend(loc="best")
    
    plt.show()

# poly_degree = int, interaction_only = True
def linear_regr(X_train, y_train, X_test, y_test, poly_degree, interaction_only, print_coef, plot, ask_user, model_result):
    
    # create more features
    poly = preprocessing.PolynomialFeatures(poly_degree, interaction_only=interaction_only)
    
    X_train = poly.fit_transform(X_train) 
    X_test = poly.fit_transform(X_test)
    (s_n, f_n) = X_train.shape
    # l_n = int(math.ceil(1.5*f_n))
    l_n = int(math.ceil(1.2*f_n))
    
        ## model_name = "SGDRegressor"
        ## model_rt_start = timeit.default_timer()
        ## regr = linear_model.SGDRegressor(penalty='elasticnet', alpha=0.01, l1_ratio=0.25, fit_intercept=True)
        ## model_rt_stop = timeit.default_timer()
        ## model_runtime = model_rt_stop - model_rt_start 
        ## # test score: 0.83
        ## model_name = "ElasticNet"
        ## model_rt_start = timeit.default_timer()
        ## regr = linear_model.ElasticNet(alpha = 0.01)
        ## model_rt_stop = timeit.default_timer()
        ## model_runtime = model_rt_stop - model_rt_start 
        if   (model == 0):
            # test score: 0.84
            alpha = 0 
            model_name = "linear_model.LinearRegression"
            regr = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
            model_rt_start = timeit.default_timer()
            regr.fit(X_train, column_or_1d(y_train) )
            model_rt_stop = timeit.default_timer()
            model_runtime = model_rt_stop - model_rt_start 
            model_result = evaluation(X_train, y_train, X_test, y_test, poly_degree, interaction_only, print_coef, plot, ask_user, 
                            model_result, model_name, model_runtime, regr, alpha)
        elif (model == 1):
            for alpha in [0.0001, 0.001, 0.01, 0.1, 1, 3, 10]:
                # test score: 0.83
                model_name = "linear_model.Lasso"
                regr_lasso = linear_model.Lasso(alpha = alpha)
                model_rt_start = timeit.default_timer()
                regr_lasso.fit(X_train, column_or_1d(y_train) )
                model_rt_stop = timeit.default_timer()
                model_runtime = model_rt_stop - model_rt_start 
                model_result = evaluation(X_train, y_train, X_test, y_test, poly_degree, interaction_only, print_coef, plot, ask_user, 
                                model_result, model_name, model_runtime, regr_lasso, alpha)
        elif (model == 2):
            for alpha in [0.0001, 0.001, 0.01, 0.1, 1, 3, 10]:
            # for alpha in [0.0000001, 0.00001, 0.001, 0.01, 0.1, 1, 3, 10, 30, 100, 300, 10**3, 10**4, 10**5]:
                # test score: 0.84
                model_name = "linear_model.Ridge"
                regr_ridge = linear_model.Ridge(alpha = alpha)
                model_rt_start = timeit.default_timer()
                regr_ridge.fit(X_train, column_or_1d(y_train) )
                model_rt_stop = timeit.default_timer()
                model_runtime = model_rt_stop - model_rt_start 
                model_result = evaluation(X_train, y_train, X_test, y_test, poly_degree, interaction_only, print_coef, plot, ask_user, 
                                model_result, model_name, model_runtime, regr_ridge, alpha)
        
_rt_stop - model_rt_start 
                        model_result = evaluation(X_train, y_train, X_test, y_test, poly_degree, interaction_only, print_coef, plot, ask_user, 
                                        model_result, model_name, model_runtime, regr, alpha)
        elif (model == 4):
            if (poly_degree <= 3):
                for alpha in [1, 10, 1000]:
                # for alpha in [0.00001]:
                    # for layer_n in [3, 7, 11]:
                    for layer_n in [7, 11]:
                    # for layer_n in [3]:
                        # test score: 0.83, runtime longer
                        model_name = "neural_network.MLPRegressor, layer = " + str(layer_n)
                        if(layer_n == 3):
                            regr = neural_network.MLPRegressor(random_state=True,hidden_layer_sizes=(l_n,l_n,l_n),alpha=alpha)
                        if(layer_n == 7):
                            # regr = neural_network.MLPRegressor(random_state=True,hidden_layer_sizes=(l_n,l_n,l_n,l_n,l_n,l_n,l_n),alpha=alpha)
                            regr = neural_network.MLPRegressor(random_state=True,hidden_layer_sizes=(l_n,l_n,l_n,l_n,l_n,l_n,l_n),alpha=alpha, learning_rate='invscaling')
                        if(layer_n == 11):
                            regr = neural_network.MLPRegressor(random_state=True,hidden_layer_sizes=(l_n,l_n,l_n,l_n,l_n,l_n,l_n,l_n,l_n,l_n,l_n),alpha=alpha)
                        model_rt_start = timeit.default_timer()
                        regr.fit(X_train, column_or_1d(y_train) )
                        model_rt_stop = timeit.default_timer()
                        model_runtime = model_rt_stop - model_rt_start 
                        model_result = evaluation(X_train, y_train, X_test, y_test, poly_degree, interaction_only, print_coef, plot, ask_user, 
                                        model_result, model_name, model_runtime, regr, alpha)
        else:
            raise SystemExit("Model selection out of range!!!")
        
    return model_result

            print("Coefficients: {}\n", regr.coefs_)
            with open("logs/log_" + log_timestr +".txt", "a") as logfile:
                logfile.write("Coefficients: {}\n".format(regr.coefs_) )
    
    
lt_timer()
    predict_test = regr.predict(X_test)
    model_rt_predict_test_stop = timeit.default_timer()
    model_runtime_predict_test = model_rt_predict_test_stop - model_rt_predict_test_start
    mse_test = float(np.mean( (predict_test - column_or_1d(y_test) ) ** 2) )
    score_test = regr.score(X_test, y_test)
    # The mean squared error
    print("Mean squared error (test): {0:.3f} \n".format( mse_test ) )
    # Explained variance score: 1 is perfect prediction
    print("Variance score (test): {0:.3f} \n".format( score_test ) )
    print("model_runtime (predict test set) = {0:.3f} (seconds) \n".format(model_runtime_predict_test))
    
    with open("logs/log_" + log_timestr +".txt", "a") as logfile:
        logfile.write("====================\n")
        logfile.write("Features polynomial degree: {} \n".format( poly_degree ) )
        logfile.write("Model: {} \n".format( model_name ) )
        logfile.write("Alpha (Regularization strength): {} \n".format( alpha ) )
        logfile.write("X_train.shape = {} \n".format(X_train.shape) )
        logfile.write("y_train.shape = {} \n".format(y_train.shape) )
        logfile.write("X_test.shape = {} \n".format(X_test.shape) )
        logfile.write("y_test.shape = {} \n".format(y_test.shape) )
        logfile.write("For training set: \n")
        logfile.write("Mean squared error (train): {0:.3f} \n".format( mse_train ) )
        logfile.write("Variance score (train): {0:.3f} \n".format( score_train ) )
        logfile.write("For test set: \n")
        logfile.write("Mean squared error (test): {0:.3f} \n".format( mse_test ) )
        logfile.write("Variance score (test): {0:.3f} \n".format( score_test ) )
        logfile.write("model_runtime (training) = {0:.3f} (seconds) \n".format(model_runtime))
        logfile.write("model_runtime (predict train set) = {0:.3f} (seconds) \n".format(model_runtime_predict_train))
        logfile.write("model_runtime (predict test set) = {0:.3f} (seconds) \n". 
    # print shape
    if (plot == True):
        plot_y_test(regr, X_test, y_test, ask_user)
    
    return model_result


# def run_fit(postfix, df_run, targets_run, features_run, poly_d_max, inter_only, print_coef, plot):
def run_fit(postfix, df_run_train, df_run_test, targets_run, features_run, poly_d_max, inter_only, print_coef, plot, ask_user):
    text = "RUNNING... df" + postfix
    print("{0:{fill}{align}16}".format(text, fill='=', align='^'))
    (X_train, y_train, X_test, y_test) = (0, 0, 0, 0) 
    # (X_train, y_train, X_test, y_test) = data_gen(df_run, targets_run, features_run, 2006, 2015, 2016, 2016)
    (X_train, y_train, X_test, y_test) = normalization(df_run_train, df_run_test, targets_run, features_run)
    # data = []
    # (data[0], data[1], data[2], data[3]) = data_gen(df_run, features_run)
    print("df{} X_train.shape = {}".format(postfix, X_train.shape))
    print("df{} y_train.shape = {}".format(postfix, y_train.shape))
    print("df{} X_test.shape = {}".format(postfix, X_test.shape))
    print("df{} y_test.shape = {}".format(postfix, y_test.shape))
    print("df_run_train target + features = {}".format(df_run_train.columns.values))
    print("=====")
    
elif (option == 'hour'):
        time_diff = time_obj - data time (time_obj.year, time_obj.month, time_obj.day, time_obj.hour, 0)
    other:
        expand SystemError ("option is not a valid thread!")
        
    return int (time_diff.total_seconds () / 60)

# do the translation
def interpolate_df (df, features):
    df_re = df
    
    print ("len (df.index) = {}". format (len (df.index)))
    
    # check all data float data and convert the data type to float64
    for col in features:
        # df [col] = df [col] .astype (float)
        temp = df [df [col] .isnull ()]
        #print (test title)
        Print ("===")
        # print (test title (n = 1))
        print ("{} type {}". format (col, df [col] .dtype))
        print ("{} type contains {} np.NaN". format (col, len (temp.index)))
        Print ("===")
    
    
    
    print ("len (df.index) = {}". format (len (df.index)))
    # can be time to use as a reference and set method = 'time'
    # df.to_csv ("df_before_interpolate.csv")
    # df [features] = df [features] .interpolate (method = 'time')
    # df.loc [:, features] = df [features] .interpolate (method = 'time')
    # somehow, df (input) will be updated or used inplace = False
    df_re.loc [:, features] = df [features] .interpolate (method = 'time', inplace = False)
    # df.to_csv ("df_after_interpolate.csv")
    #print ("df =")
    #print (df)
    
    # capture real nan values
    df_nan_interpolate = df.loc [df_nan.index.values]
    print ("len (df_nan_interpolate.index) = {}". format (len (df_nan_interpolate.index)))
    df_nan_interpolate.to_csv ("df_nan_interpolate.csv")
    
    if (df_re.notnull (). all (axis = 1) .all (axis = 0)):
        print ("CHECK: No value in df_re.")
        
    replace df_re

    df_train = df.loc [(df.index> data_start) & (df.index <= data_end) ,:]]
    
    # do interpolate on training set only
    df_train = interpolate_df (df_train, features)
    df_train.to_csv ('df_train_clean.csv')
    
    X_train = df_train [features]
    y_train = df_train [target]
    
    # configure test data
    data_start = time (data_test_yr_start, 1, 1, 0, 0, 0)
    data_end = time (data_test_yr_end, 12, 31, 23, 59, 59)
    df_test = df.loc [(df.index> data_start) & (df.index <= data_end) ,:]]
    
    # drops the number lines of the NaN test set
    (irowu_old, col_old) = df_test.shape
    print ("Before you draw the NaN number of the test set, df_test.shape = {}". format (df_test.shape))
    df_test = df_test [df_test.notnull (). all (axis = 1)]
    (row, column) = df_test.shape
    print ("After dragging the NaN number of the test set, df_test.shape = {}". format (df_test.shape))
    print ("Drop Level = {0: .2f}". format (float (1 - (line / line_knee))))
    
    df_test.to_csv ('df_test_clean.csv')
    X_test = df_test [features]
    y_test = df_test [stones]
    
    # familiarity and training scale / test set
    # use robust_scaler to protect misleading merchants
    # scaler = processing StandardScaler ()
    # use robust_scaler to protect misleading merchants
    scale = advancement.RebustScaler ()
    X_train = scaler.fit_transform (X_train)
    X_test = scaler.transform (X_test)
    
    return (X_train, y_train, X_test, y_test)
    
def normalization (df_train, df_test, targets, features):
    
    # use robust_scaler to protect misleading merchants
    scale = advancement.RebustScaler ()
    X_train = scaler.fit_transform (X_train)
    X_test = scaler.transform (X_test)
    
    return (X_train, y_train, X_test, y_test)

# structure y_test
def plot_y_test (regr, X_test, y_test, ask_user):
    (r_test, c_test) = X_test.shape
    
    # for i in range (c_test):
    # plt. scatter (X_test [:, i], y_test)
    # plt.plot (X_test [:, i], regr.predict (X_test), color = 'blue', linewidth = 3)
    
    y_predict = regr.predict (X_test)
    #print ("==> y_test type = {}". format (type (y_test)))
    #print ("y_test.index = {}". format (y_test.index))
    #print ("y_test = {}". format (y_test))
    #print ("y_predict = {}". format (y_predict))
    df_plot = y_test
    #print (df_plot)
    #print ("DATE")
    #print ("# # # # # # # # # # # # # # # # # # #
    df_plot = df_plot.reset_index (level = ['DATE'])
    df_plot.loc [:, 'predict_temp_C'] = y_cognize
    # go back green DATE time 1day_later
    df_plot.loc [:, "raw_DATE"] = df_plot ['DATE']. apply (lambda time_obj: time_obj + relativedelta (days = 1))
    df_plot.rename (columns = {'1days_later_temp_C': 'raw_temp_C', 'DATE': 'label_DATE'}, inplace = True)
    
    df_plot = df_plot.set_index ("green_DATE")
    #print (df_plot)
    #print ("# # # # # # # # # # # # # # # # # #
    
    # the time range for the default layout
    icebo_yr = 2016
    structure_month = 10
    icebo_day = 5
    duration = 10

    model_re = {}
    
    # for poly_d in range(1, poly_d_max+1):
    for poly_d in range(1, poly_d_max+1):
        model_re = linear_regr(X_train, y_train, X_test, y_test, poly_degree = poly_d, 
            interaction_only = inter_only, print_coef = print_coef, plot = plot, ask_user = ask_user, model_result = model_re)
    return model_re

elif (option == 'hour'):
        time_diff = time_obj - data time (time_obj.year, time_obj.month, time_obj.day, time_obj.hour, 0)
    other:
        expand SystemError ("option is not a valid thread!")
        
    return int (time_diff.total_seconds () / 60)

# do the translation
def interpolate_df (df, features):
    df_re = df
    
    print ("len (df.index) = {}". format (len (df.index)))
    
    # check all data float data and convert the data type to float64
    for col in features:
        # df [col] = df [col] .astype (float)
        temp = df [df [col] .isnull ()]
        #print (test title)
        Print ("===")
        # print (test title (n = 1))
        print ("{} type {}". format (col, df [col] .dtype))
        print ("{} type contains {} np.NaN". format (col, len (temp.index)))
        Print ("===")
    
    
    
    print ("len (df.index) = {}". format (len (df.index)))
    # can be time to use as a reference and set method = 'time'
    # df.to_csv ("df_before_interpolate.csv")
    # df [features] = df [features] .interpolate (method = 'time')
    # df.loc [:, features] = df [features] .interpolate (method = 'time')
    # somehow, df (input) will be updated or used inplace = False
    df_re.loc [:, features] = df [features] .interpolate (method = 'time', inplace = False)
    # df.to_csv ("df_after_interpolate.csv")
    #print ("df =")
    #print (df)
    
    # capture real nan values
    df_nan_interpolate = df.loc [df_nan.index.values]
    print ("len (df_nan_interpolate.index) = {}". format (len (df_nan_interpolate.index)))
    df_nan_interpolate.to_csv ("df_nan_interpolate.csv")
    
    if (df_re.notnull (). all (axis = 1) .all (axis = 0)):
        print ("CHECK: No value in df_re.")
        
    replace df_re

    df_train = df.loc [(df.index> data_start) & (df.index <= data_end) ,:]]
    
    # do interpolate on training set only
    df_train = interpolate_df (df_train, features)
    df_train.to_csv ('df_train_clean.csv')
    
    X_train = df_train [features]
    y_train = df_train [target]
    
    # configure test data
    data_start = time (data_test_yr_start, 1, 1, 0, 0, 0)
    data_end = time (data_test_yr_end, 12, 31, 23, 59, 59)
    df_test = df.loc [(df.index> data_start) & (df.index <= data_end) ,:]]
    
    # drops the number lines of the NaN test set
    (irowu_old, col_old) = df_test.shape
    print ("Before you draw the NaN number of the test set, df_test.shape = {}". format (df_test.shape))
    df_test = df_test [df_test.notnull (). all (axis = 1)]
    (row, column) = df_test.shape
    print ("After dragging the NaN number of the test set, df_test.shape = {}". format (df_test.shape))
    print ("Drop Level = {0: .2f}". format (float (1 - (line / line_knee))))
    
    df_test.to_csv ('df_test_clean.csv')
    X_test = df_test [features]
    y_test = df_test [stones]
    
    # familiarity and training scale / test set
    # use robust_scaler to protect misleading merchants
    # scaler = processing StandardScaler ()
    # use robust_scaler to protect misleading merchants
    scale = advancement.RebustScaler ()
    X_train = scaler.fit_transform (X_train)
    X_test = scaler.transform (X_test)
    
    return (X_train, y_train, X_test, y_test)
    
def normalization (df_train, df_test, targets, features):
    
    # use robust_scaler to protect misleading merchants
    scale = advancement.RebustScaler ()
    X_train = scaler.fit_transform (X_train)
    X_test = scaler.transform (X_test)
    
    return (X_train, y_train, X_test, y_test)

# structure y_test
def plot_y_test (regr, X_test, y_test, ask_user):
    (r_test, c_test) = X_test.shape
    
    # for i in range (c_test):
    # plt. scatter (X_test [:, i], y_test)
    # plt.plot (X_test [:, i], regr.predict (X_test), color = 'blue', linewidth = 3)
    
    y_predict = regr.predict (X_test)
    #print ("==> y_test type = {}". format (type (y_test)))
    #print ("y_test.index = {}". format (y_test.index))
    #print ("y_test = {}". format (y_test))
    #print ("y_predict = {}". format (y_predict))
    df_plot = y_test
    #print (df_plot)
    #print ("DATE")
    #print ("# # # # # # # # # # # # # # # # # # #
    df_plot = df_plot.reset_index (level = ['DATE'])
    df_plot.loc [:, 'predict_temp_C'] = y_cognize
    # go back green DATE time 1day_later
    df_plot.loc [:, "raw_DATE"] = df_plot ['DATE']. apply (lambda time_obj: time_obj + relativedelta (days = 1))
    df_plot.rename (columns = {'1days_later_temp_C': 'raw_temp_C', 'DATE': 'label_DATE'}, inplace = True)
    
    df_plot = df_plot.set_index ("green_DATE")
    #print (df_plot)
    #print ("# # # # # # # # # # # # # # # # # #
    
    # the time range for the default layout
    icebo_yr = 2016
    structure_month = 10
    icebo_day = 5
    duration = 10
    
    t3 = t1
    t3.loc[:, new_target] = t2[new_target]
    # df_time_train = t3['2006':'2014']
    range_start = datetime(train_yr_start, 1, 1, 0, 0, 0)
    range_end   = datetime(train_yr_start, 1, 1, 0, 0, 0) + relativedelta(years=train_years)
    
    df_time_train = t3[range_start.strftime('%Y-%m-%d %H:%M:%S') : range_end.strftime('%Y-%m-%d %H:%M:%S')] 
    print("df_time_train.shape = {}".format(df_time_train.shape))
    df_time_train.loc[:, new_target] = df_time_train[new_target].interpolate(method='time')
    
    # df_time_test = t3['2015'] 
    range_start = datetime(test_yr_start, s_month, s_day, 0, 0, 0)
    range_end   = datetime(test_yr_start, s_month, s_day, 0, 0, 0) + relativedelta(days=365)
    # range_end   = datetime(2017, 3, 05, 0, 0, 0)
    # range_end   = datetime(2017, 3, 05, 0, 0, 0)
    
    df_time_test = t3[range_start.strftime('%Y-%m-%d %H:%M:%S') : range_end.strftime('%Y-%m-%d %H:%M:%S')] 
    print("df_time_test.shape = {}".format(df_time_train.shape))
    
    # drop NaN number rows of test set
    (row_old, col_old) = df_time_test.shape
    print("Before drop NaN number of test set, df_time_test.shape = {}".format(df_time_test.shape))
    df_time_test = df_time_test[ df_time_test.notnull().all(axis=1) ]
    (row, col) = df_time_test.shape
    print("After drop NaN number of test set, df_time_test.shape = {}".format(df_time_test.shape))
    print("Drop rate = {0:.2f} ".format(float(1 - (row/row_old)) ) )
    
    print("===================== \n")
    print("### Experiment = {} \n".format( experiment ) )
    print("new_target = {} \n".format( new_target ) )
    print("new_features = {} \n".format( new_features ) )
    with open("logs/log_" + log_timestr +".txt", "a") as logfile:
        logfile.write("===================== \n")
        logfile.write("### Experiment = {} \n".format( experiment ) )
        logfile.write("new_target = {} \n".format( new_target ) )
        logfile.write("new_features = {} \n".format( new_features ) )
False, ask_user=False)
    
    print("### Experiment = {} \n".format( experiment ) )
    # print("model_re = {}".format(model_re))
