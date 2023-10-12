import pandas as pd
import numpy as np
import pvlib
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR


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


class tree:
    def __init__(self):
        self.rf = None
        self.dt = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def dt_depth(self, depth):
        self.dt = DecisionTreeRegressor(max_depth=depth)


    def rf_fit(self, max_depth = 5, n_estimators = 100, learning_rate = 0.01, num_leves = 8, Grid=False):
        """
        x_train : 학습할 데이터
        y_train : target
        max depth : decision tree의 최대 깊이 
        n_estimators : 생성할 tree의 개수
        learning_rate : 학습률
        num_leves : 최대 leaf의 개수
        
        Grid : GridSearchCV 사용 여부
        Grid가 True일 경우, GridSearchCV를 사용하여 최적의 파라미터를 찾는다.
        이때 parameter를 list으로 설정

        Grid가 False인 경우, 설정된 상수 값으로 모델 생성
        """
        self.rf = RandomForestRegressor()

        if Grid:
            param_grid = {
                'max_depth': max_depth,
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'num_leaves': num_leves
            }
            grid_cv_rf = GridSearchCV(self.rf, param_grid=param_grid,
                        cv=5, n_jobs=-1)
            model = grid_cv_rf.fit(self.x_train, self.y_train)

        else:
            model = self.rf.fit(self.x_train, self.y_train, max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leves)
        
        return model
    
    def rf_pred(self, x_test):
        """
        x_test : 예측할 데이터
        """
        return self.rf.predict(x_test)
    
    def ada_fit(self, max_depth = 5, n_estimators = 100, learning_rate = 0.01, Grid=False):
        """
        x_train : 학습할 데이터
        y_train : target
        max depth : decision tree의 최대 깊이 
        n_estimators : 생성할 tree의 개수
        learning_rate : 학습률
        
        Grid : GridSearchCV 사용 여부
        Grid가 True일 경우, GridSearchCV를 사용하여 최적의 파라미터를 찾는다.
        이때 parameter를 list으로 설정

        Grid가 False인 경우, 설정된 상수 값으로 모델 생성
        """
        self.ada = AdaBoostRegressor()

        if Grid:
            if max_depth.dtype & n_estimators.dtype & learning_rate.dtype != list:
                raise ValueError("one vlaue must be list")
            param_grid = {
                'max_depth': max_depth,
                'n_estimators': n_estimators,
                'learning_rate': learning_rate
            }
            grid_cv_ada = GridSearchCV(self.ada, param_grid=param_grid,
                        cv=5, n_jobs=-1)
            model = grid_cv_ada.fit(self.x_train, self.y_train)

        else:
            model = self.ada.fit(self.x_train, self.y_train, max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate)
        
        return model
    


