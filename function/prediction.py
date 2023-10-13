import pandas as pd
import numpy as np
import pvlib
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from tensorflow.keras import layers, models
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import xgboost


def clear_sky_model_pred(longitude, latitude, capacity, weather):
    """
    input :
        longitude : 경도
        latitude : 위도
        capacity : 용량(단위 : w)
        weather : 날씨 데이터 (index가 날짜인 데이터)
    output :
        cs : ghi, dni, dhi
        generation : 발전량
    """
    location = pvlib.location.Location(latitude, longitude, tz='Asia/Seoul')
    start_date = weather.index.min()
    end_date = weather.index.max()

    # 시간대가 Asia/Seoul로 설정되어 있는지 확인하고, 아니라면 설정한다.
    if weather.index.tz is None or weather.index.tz.zone != 'Asia/Seoul':
        weather = weather.tz_localize('Asia/Seoul')

    times = pd.date_range(start=start_date, end=end_date, freq='1H', tz='Asia/Seoul')
    solpos = location.get_solarposition(times=times)
    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    airmass = pvlib.atmosphere.get_relative_airmass(solpos['apparent_zenith'])
    pressure = pvlib.atmosphere.alt2pres(location.altitude)
    am_abs = pvlib.atmosphere.get_absolute_airmass(airmass, pressure) 
    tl = pvlib.clearsky.lookup_linke_turbidity(times, latitude, longitude)

    solis_clearsky = pvlib.clearsky.simplified_solis(solpos['apparent_zenith'], am_abs, tl)
    cs = location.get_clearsky(times, model='simplified_solis')
    
    # cs의 시간대를 None으로 설정
    cs = cs.tz_localize(None)

    system = pvlib.pvsystem.PVSystem(surface_tilt=30, surface_azimuth=180,
                                    module_parameters={'pdc0': capacity, 'gamma_pdc': -0.004}, 
                                    inverter_parameters={'pdc0': capacity},
                                    modules_per_string=1, strings_per_inverter=1,
                                    temperature_model_parameters={'a': -3.56, 'b': -0.075, 'deltaT': 3})
    mc = pvlib.modelchain.ModelChain(system, location, spectral_model='no_loss', aoi_model='no_loss')

    mc.run_model(pd.concat([solis_clearsky, weather], axis=1))

    return cs, pd.DataFrame(mc.results.ac)


class model:
    def __init__(self, x_train, y_train, x_test, y_test, x_scaler_type=None, y_scaler_type=None):
        self.rf = None
        self.dt = None
        self.lightgbm = None
        self.xgboost = None
        self.lstm = None
        self.dnn = None
        self.svr = None
        self.et = None

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_scaler = self._get_scaler(x_scaler_type) if x_scaler_type else None
        self.y_scaler = self._get_scaler(y_scaler_type) if y_scaler_type else None


    def _get_scaler(self, scaler_type):
        if scaler_type.lower() == 'minmax':
            return MinMaxScaler()
        elif scaler_type.lower() == 'standard':
            return StandardScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")

    def apply_scaling(self):
        if self.x_scaler:
            self.x_train = self.x_scaler.fit_transform(self.x_train)
            self.x_test = self.x_scaler.transform(self.x_test)
            
        if self.y_scaler:
            self.y_train = self.y_scaler.fit_transform(self.y_train.reshape(-1, 1)).flatten()
            self.y_test = self.y_scaler.transform(self.y_test.reshape(-1, 1)).flatten()

    def rf_fit(self, max_depth = [None], 
               n_estimators = [100], 
               min_samples_leaf = [1], 
               min_samples_split = [2],
               criterion = ['squared_error'],
               max_leaf_nodes = [None],
               bootstrap = [True], 
               Grid=False):
        """

        """

        if Grid:
            param_grid = {
                'max_depth': max_depth,
                'n_estimators': n_estimators,
                'min_samples_leaf': min_samples_leaf,
                'min_samples_split': min_samples_split,
                'criterion': criterion,
                'bootstrap': bootstrap,
            }
            self.rf = RandomForestRegressor()
            grid_cv_rf = GridSearchCV(self.rf, param_grid=param_grid,
                        cv=5, n_jobs=-1)
            model = grid_cv_rf.fit(self.x_train, self.y_train)

        else:
            self.rf = RandomForestRegressor(max_depth=max_depth[0],
                                            n_estimators = n_estimators[0], 
                                            min_samples_leaf = min_samples_leaf[0], 
                                            min_samples_split = min_samples_split[0],
                                            criterion = criterion[0],
                                            bootstrap=bootstrap[0])
            model = self.rf.fit(self.x_train, self.y_train)
        
        return model
    
    def rf_pred(self, x_test):
        """
        x_test : 예측할 데이터
        """
        return self.rf.predict(x_test)
    
    def ada_fit(self, max_depth = [5], n_estimators = [100], learning_rate = [0.01], loss = ['linear'], Grid=False):
        """

        """
        if Grid:
            param_grid = {
                'base_estimator': [DecisionTreeRegressor(max_depth=md) for md in max_depth],
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'loss': loss
            }
            self.ada = AdaBoostRegressor()
            grid_cv_ada = GridSearchCV(self.ada, param_grid=param_grid,
                        cv=5, n_jobs=-1)
            model = grid_cv_ada.fit(self.x_train, self.y_train)
            self.ada = model.best_estimator_

        else:
            # If Grid is not True, use the first value from each list of parameters
            self.ada = AdaBoostRegressor(
                base_estimator=DecisionTreeRegressor(max_depth=max_depth),
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                loss=loss
            )
            model = self.ada.fit(self.x_train, self.y_train)
        
        return model

    def ada_pred(self):
        """
        input:
            None

        output:
            adaboost prediction
        """
        return self.ada.predict(self.x_test)
    


    def dt_fit(self, max_depth = [3], min_samples_leaf = [1], min_samples_split = [2], Grid=False):
        """

        """
        if Grid:
            param_grid = {
                'max_depth': max_depth,
                'min_samples_leaf': min_samples_leaf,
                'min_samples_split': min_samples_split
            }
            self.dt = DecisionTreeRegressor()
            grid_cv_dt = GridSearchCV(self.dt, param_grid=param_grid,
                        cv=5, n_jobs=-1)
            model = grid_cv_dt.fit(self.x_train, self.y_train)

        else:
            self.dt = DecisionTreeRegressor(max_depth = max_depth[0], 
                                            min_samples_leaf = min_samples_leaf[0], 
                                            min_samples_split = min_samples_split[0])
            model = self.dt.fit(self.x_train, self.y_train)
        
        return model
    
    def dt_pred(self):
        """
        input:
            None

        output:
            decision tree prediction
        """
        return self.dt.predict(self.x_test)
    
    def lightgbm_fit(self, max_depth = [3], n_estimators = [100], learning_rate = [0.01], Grid=False):
        """

        """
        if Grid:
            param_grid = {
                'max_depth': max_depth,
                'n_estimators': n_estimators,
                'learning_rate': learning_rate
            }
            self.lightgbm = xgboost.XGBRegressor()
            grid_cv_lightgbm = GridSearchCV(self.lightgbm, param_grid=param_grid,
                        cv=5, n_jobs=-1)
            model = grid_cv_lightgbm.fit(self.x_train, self.y_train)

        else:
            self.lightgbm = xgboost.XGBRegressor(max_depth = max_depth[0], 
                                            n_estimators = n_estimators[0], 
                                            learning_rate = learning_rate[0])
            model = self.lightgbm.fit(self.x_train, self.y_train)
        
        return model
    
    def lightgbm_pred(self):
        """
        input:
            None

        output:
            lightgbm prediction
        """
        return self.lightgbm.predict(self.x_test)
    
    def xgboost_fit(self, max_depth = [3], n_estimators = [100], learning_rate = [0.01], Grid=False):
        """

        """
        if Grid:
            param_grid = {
                'max_depth': max_depth,
                'n_estimators': n_estimators,
                'learning_rate': learning_rate
            }
            self.xgboost = xgboost.XGBRegressor()
            grid_cv_xgboost = GridSearchCV(self.xgboost, param_grid=param_grid,
                        cv=5, n_jobs=-1)
            model = grid_cv_xgboost.fit(self.x_train, self.y_train)

        else:
            self.xgboost = xgboost.XGBRegressor(max_depth = max_depth[0], 
                                            n_estimators = n_estimators[0], 
                                            learning_rate = learning_rate[0])
            model = self.xgboost.fit(self.x_train, self.y_train)
        
        return model
    
    def xgboost_pred(self):
        """
        input:
            None

        output:
            xgboost prediction
        """
        return self.xgboost.predict(self.x_test)
    
    def extra_tree_fit(self, criterion='mse',
                       max_depth=[None], n_estimators=[100],
                       min_samples_leaf=[1], min_samples_split=[2],
                       max_leaf_nodes=[None], bootstrap=False, Grid=False):
        """

        """
        if Grid:
            param_grid = {
                'max_depth': max_depth,
                'n_estimators': n_estimators,
                'min_samples_leaf': min_samples_leaf,
                'min_samples_split': min_samples_split,
                'max_leaf_nodes': max_leaf_nodes
            }
            self.et = ExtraTreesRegressor(bootstrap=bootstrap, criterion=criterion)
            grid_cv_et = GridSearchCV(self.et, param_grid=param_grid, cv=5, n_jobs=-1)
            model = grid_cv_et.fit(self.x_train, self.y_train)

        else:
            self.et = ExtraTreesRegressor(criterion=criterion, max_depth=max_depth[0],
                                          n_estimators=n_estimators[0],
                                          min_samples_leaf=min_samples_leaf[0],
                                          min_samples_split=min_samples_split[0],
                                          max_leaf_nodes=max_leaf_nodes[0],
                                          bootstrap=bootstrap)
            model = self.et.fit(self.x_train, self.y_train)

        return model
    
    def extrea_tree_pred(self):
        """

        """
        return self.et.predict(self.x_test)
        
    def lstm_train(self, node, epoch, depth, batch_size, activation, loss, optimizer, drop=None):
        self.lstm = Sequential()
        self.lstm.add(LSTM(node, input_shape=(self.x_train.shape[1], self.x_train.shape[2])))

        for i in range(depth):
            node = max(1, node // 2)  # Decreasing the number of nodes
            self.lstm.add(layers.LSTM(node, activation=activation))
            if drop:
                self.lstm.add(layers.Dropout(drop))

        output_nodes = self.y_train.shape[1] if len(self.y_train.shape) > 1 else 1  # Adjusting the output nodes based on y's shape
        self.lstm.add(layers.Dense(output_nodes))
        self.lstm.compile(loss=loss, optimizer=optimizer)
        lstm_his = self.lstm.fit(self.x_train, self.y_train, epochs=epoch, batch_size=batch_size, verbose=0)

        return lstm_his.history['loss']
    
    def lstm_pred(self):
        """
        input:
            None

        output:
            lstm prediction
        """
        return self.lstm.predict(self.x_test)
    

    def dnn_train(self, node, epoch, depth, batch_size, activation, loss, optimizer, drop = False):
        """
        input:
            node : dnn의 node 개수
            epoch : 학습 횟수
            depth : layer 개수
            batch_size : batch size
            activation : activation function
            loss : loss function
            optimizer : optimizer

        return:
            dnn history
        """
        self.dnn = Sequential()
        self.dnn.add(layers.Dense(node, input_shape=(self.x_train.shape[1],)))

        for i in range(depth):
            node = max(1, node // 2)  # Decreasing the number of nodes
            self.dnn.add(layers.Dense(node, activation=activation))
            if drop:
                self.dnn.add(layers.Dropout(drop))

        output_nodes = self.y_train.shape[1] if len(self.y_train.shape) > 1 else 1  # Adjusting the output nodes based on y's shape
        self.dnn.add(layers.Dense(output_nodes))
        self.dnn.compile(loss=loss, optimizer=optimizer)
        dnn_his = self.dnn.fit(self.x_train, self.y_train, epochs=epoch, batch_size=batch_size, verbose=0)

        return dnn_his.history['loss']
    
    def dnn_pred(self):
        """
        input:
            None

        output:
            dnn prediction
        """
        return self.dnn.predict(self.x_test)
    


    def svr_fit(self, kernel, C, epsilon, gamma, Grid=False):
        """
        input:
            kernel : kernel
            C : C
            epsilon : epsilon
            gamma : gamma

        return:
            svr model
        """
        
        if Grid:
            param_grid = {
                'kernel': kernel,
                'C': C,
                'epsilon': epsilon,
                'gamma': gamma
            }
            self.svr = SVR()
            grid_cv_svr = GridSearchCV(self.svr, param_grid=param_grid,
                        cv=5, n_jobs=-1)
            model = grid_cv_svr.fit(self.x_train, self.y_train)

        else:
            self.svr = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
            model = self.svr.fit(self.x_train, self.y_train, kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
        
        return model
    
    def svr_pred(self):
        """
        input:
            None

        output:
            svr prediction
        """
        return self.svr.predict(self.x_test)