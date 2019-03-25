from fmipp.export.FMIAdapterV2 import FMIAdapterV2
import pandapower as pp
import pandas as pd
import numpy as np
import os

class PandapowerFMUClass(FMIAdapterV2):
	"""
	implements methods for initialization and time steps for pandapower
	as pandapower does not specifically support time domain simulation additional methods had to be implemented:
		- use time series data for specific variables in pandapower networks
		- interpolate/extrapolate time series data in case of missing time
		- write results for all variables in a single result file
		- set FMU inputs and outputs
	"""

	# methods needed to implement FMIAdapterV2 class
	def init(self, currentCommunicationPoint):
		"""
		Initialize the FMU (definition of input/output variables and parameters, enforce step size).
		"""

		### SAVE START TIME
		self._t_start = currentCommunicationPoint

		### LOAD PANDAPOWER NETWORK FROM FILE ###
		for file in os.listdir(os.getcwd()):  # search for first file with ending '.p'
			if file.endswith('.p'):
				self.net = pp.from_pickle(file)  # load net from pickle file
		if not hasattr(self, 'net'):
			raise RuntimeError('Loading pandapower network from file failed')

		### DEFINE INPUTS OF FMU ###
		self._initialize_fmi_inputs()

		### LOADING PROFILES ###
		self._load_profiles()

		### RUN INITIAL POWERFLOW ###
		pp.runpp(self.net)  # run one powerflow to initialize the results dataframes used by the next powerflow as init

		### WRITE HEADER TO RESULT FILE ###
		df_results = get_pp_results(self.net, 0.0).iloc[
					 0:0]  # create empty dataframe for results with valid column names
		df_results.to_csv(self.net.result_file)

	def doStep(self, currentCommunicationPoint, communicationStepSize):
		"""
		Make a simulation step.
		"""
		### DEFINE ALL TIMES WHERE POWERFLOW IS DONE ###
		# current simulation time is always included
		sim_steps = [currentCommunicationPoint]
		# check if additional power flows should be done every time step
		if hasattr(self.net, 'time_step'):
			# get all multiples of time_step during currentCommunicationPoint and currentCommunicationPoint+communicationStepSize
			sim_steps_add = [t_sym for t_sym in np.arange(
				currentCommunicationPoint + (
				self.net.time_step - (currentCommunicationPoint - self._t_start) % self.net.time_step),
				currentCommunicationPoint + communicationStepSize, self.net.time_step)]
			# combine all times
			sim_steps = np.concatenate((sim_steps, sim_steps_add), axis=None)

		### GET AND SET ALL INPUTS OF FMU
		self._get_and_set_fmi_inputs()

		### LOOP OVER ALL TIME STEPS ###
		for t_sym in sim_steps:
			### SET PROFILES ###
			self._set_profile_values(t_sym)

			### RUN POWERFLOW ###
			pp.runpp(self.net, init='results', recycle={'ppc': True,
														'Ybus': True})  # run powerflow; initialize load flow using results
			# from last load flow (error raised if no results dataframes); recycle data; ppc is taken from net[“_ppc”]
			# and gets updated instead of reconstructed entirely & the admittance matrix (Ybus, Yf, Yt) is taken from
			# ppc[“internal”] and not reconstructed

			### WRITE RESULTS ###
			with open(self.net.result_file, 'a') as result_file:
				result_file.write(
					str(t_sym) + ',' + get_pp_results(self.net, t_sym).iloc[
						0].to_string(index=False, header=True).replace('\n', ',').replace(' ', '') + '\n')

		### SET FMU OUTPUTS ###
		self._set_outputs()

	# private methods
	def _initialize_fmi_inputs(self):
		"""
		initializes all FMI inputs as specified directly in the pandapower network objects 'fmi_input_vars'/
		'fmi_output_vars'/'fmi_parameters'
		the input/output/parameter names must follow the naming convention 'class.component.parameter'
			- class: must be the name of a pandas dataframe in the pandapower network, e.g., 'load'; also results
			dataframes can be used, e.g., 'res_load'
			- component: this is the name of the component, e.g., 'static generator 1'; be aware that this name must
			be unique within each class
			- parameter: the name of the parameter of the component that, e.g., 'p_mw'
		"""
		# define inputs
		if hasattr(self.net, 'fmi_input_vars'):
			if 'Real' in self.net.fmi_input_vars.keys():
				self.defineRealInputs(*self.net.fmi_input_vars['Real'])
			if 'Integer' in self.net.fmi_input_vars.keys():
				self.defineIntegerInputs(*self.net.fmi_input_vars['Integer'])
			if 'Boolean' in self.net.fmi_input_vars.keys():
				self.defineBooleanInputs(*self.net.fmi_input_vars['Boolean'])
			if 'String' in self.net.fmi_input_vars.keys():
				self.defineStringInputs(*self.net.fmi_input_vars['String'])
		# define outputs
		if hasattr(self.net, 'fmi_output_vars'):
			if 'Real' in self.net.fmi_output_vars.keys():
				self.defineRealOutputs(*self.net.fmi_output_vars['Real'])
			if 'Integer' in self.net.fmi_output_vars.keys():
				self.defineRealOutputs(*self.net.fmi_output_vars['Integer'])
			if 'Boolean' in self.net.fmi_output_vars.keys():
				self.defineRealOutputs(*self.net.fmi_output_vars['Boolean'])
			if 'String' in self.net.fmi_output_vars.keys():
				self.defineRealOutputs(*self.net.fmi_output_vars['String'])
		# define parameters
		if hasattr(self.net, 'fmi_parameters'):
			if 'Real' in self.net.fmi_parameters.keys():
				self.defineRealParameters(*self.net.fmi_parameters['Real'])
			if 'Integer' in self.net.fmi_parameters.keys():
				self.defineIntegerParameters(*self.net.fmi_parameters['Integer'])
			if 'Boolean' in self.net.fmi_parameters.keys():
				self.defineBooleanParameters(*self.net.fmi_parameters['Boolean'])
			if 'String' in self.net.fmi_parameters.keys():
				self.defineStringParameters(*self.net.fmi_parameters['String'])

	def _load_profiles(self):
		"""
		reads all time-series profiles from files to pandas series as specified directly in pandapower network object
		attribute 'profile'; however series can also be passed directly using this object, then they are loaded
		together with the pandapower network object
		"""
		if hasattr(self.net, 'profiles'):  # check if profiles are present in network
			for index, row in self.net.profiles.iterrows():
				if not isinstance(row['profile'], pd.Series):  # check if profile is already given
					row['profile'] = \
						pd.read_csv(row['file'], header=None, index_col=0, usecols=[0, int(row['column'])])[
							int(row['column'])]  # load all profiles to dataframe series

	def _set_profile_values(self, time):
		"""
		for all time dependent variables specified in the pandapower network attribute 'profiles' the current value is
		taken and the corresponding parameter is set to this value; if no exact time matches the value is linearly
		interpolated if it is within the profile series and otherwise the first/last value is used as extrapolation
		:param time: current simulation time
		:type time: float
		"""
		if hasattr(self.net, 'profiles'):  # check if profiles are present in network
			for index, row in self.net.profiles.iterrows():
				value = get_interpolated_row(row['profile'], time)  # interpolate if time does not exist in series
				getattr(self.net, row['class'])[row['parameter']][
					getattr(self.net, row['class']).index[
						getattr(self.net, row['class'])['name'] == row['component']]] = value

	def _get_and_set_fmi_inputs(self):
		"""
		reads the current inputs of the FMU and sets the values to the corresponding variables in the pandapower network
		"""
		### READ ALL FMU INPUTS ###
		if hasattr(self.net, 'fmi_input_vars'):
			fmi_inputs = {}
			fmi_inputs['Real'] = self.getRealInputValues()
			fmi_inputs['Integer'] = self.getIntegerInputValues()
			fmi_inputs['Boolean'] = self.getBooleanInputValues()
			fmi_inputs['String'] = self.getStringInputValues()

		### SET FMU INPUTS ###
		if hasattr(self.net, 'fmi_input_vars'):
			for type in fmi_inputs.keys():
				for key, val in fmi_inputs[type].items():
					classname = key.split('.')[0]  # name must correspond to format class.component.parameter
					componentname = key.split('.')[1]
					parametername = key.split('.')[2]
					if classname.startswith(
							'res_'):  # get name of components from corresponding component definition dataframe
						ix = getattr(self.net, classname).index[
							getattr(self.net, classname.split('res_')[1])['name'] == componentname]
					else:
						ix = getattr(self.net, classname).index[getattr(self.net, classname)['name'] == componentname]
					getattr(self.net, classname)[parametername][ix] = val

	def _set_outputs(self):
		"""
		passes the current values of the pandapower network variables to their corresponding FMU outputs
		"""
		if hasattr(self.net, 'fmi_output_vars'):
			if 'Real' in self.net.fmi_output_vars.keys():
				outputs = {}
				for key in self.net.fmi_output_vars['Real']:
					classname = key.split('.')[0]  # name must correspond to format class.component.parameter
					componentname = key.split('.')[1]
					parametername = key.split('.')[2]
					if classname.startswith(
							'res_'):  # get name of components from corresponding component definition dataframe
						ix = getattr(self.net, classname).index[
							getattr(self.net, classname.split('res_')[1])['name'] == componentname]
					else:
						ix = getattr(self.net, classname).index[getattr(self.net, classname)['name'] == componentname]
					outputs[key] = getattr(self.net, classname)[parametername][ix][0]
				self.setRealOutputValues(outputs)
			if 'Integer' in self.net.fmi_output_vars.keys():
				outputs = {}
				for key in self.net.fmi_output_vars['Integer']:
					classname = key.split('.')[0]  # name must correspond to format class.component.parameter
					componentname = key.split('.')[1]
					parametername = key.split('.')[2]
					if classname.startswith(
							'res_'):  # get name of components from corresponding component definition dataframe
						ix = getattr(self.net, classname).index[
							getattr(self.net, classname.split('res_')[1])['name'] == componentname]
					else:
						ix = getattr(self.net, classname).index[getattr(self.net, classname)['name'] == componentname]
					outputs[key] = getattr(self.net, classname)[parametername][ix][0]
				self.setIntegerOutputValues(outputs)
			if 'Boolean' in self.net.fmi_output_vars.keys():
				outputs = {}
				for key in self.net.fmi_output_vars['Boolean']:
					classname = key.split('.')[0]  # name must correspond to format class.component.parameter
					componentname = key.split('.')[1]
					parametername = key.split('.')[2]
					if classname.startswith(
							'res_'):  # get name of components from corresponding component definition dataframe
						ix = getattr(self.net, classname).index[
							getattr(self.net, classname.split('res_')[1])['name'] == componentname]
					else:
						ix = getattr(self.net, classname).index[getattr(self.net, classname)['name'] == componentname]
					outputs[key] = getattr(self.net, classname)[parametername][ix][0]
				self.setBooleanOutputValues(outputs)
			if 'String' in self.net.fmi_output_vars.keys():
				outputs = {}
				for key in self.net.fmi_output_vars['String']:
					classname = key.split('.')[0]  # name must correspond to format class.component.parameter
					componentname = key.split('.')[1]
					parametername = key.split('.')[2]
					if classname.startswith(
							'res_'):  # get name of components from corresponding component definition dataframe
						ix = getattr(self.net, classname).index[
							getattr(self.net, classname.split('res_')[1])['name'] == componentname]
					else:
						ix = getattr(self.net, classname).index[getattr(self.net, classname)['name'] == componentname]
					outputs[key] = getattr(self.net, classname)[parametername][ix][0]
				self.setStringOutputValues(outputs)


### HELPER FUNCTIONS

def get_interpolated_row(df, time):
	"""
	for float types it linearly interpolates between two pandas dataframe rows and returns the interpolated row, for all
	other types it takes the nearest value; extrapolates the dataframe using first/last row
	:param df: pandas dataframe or series containing time series profiles
	:type df: pandas.DataFrame or pandas.Series
	:param time: current simulation time
	:type time: float
	:return: linearly interpolated/from first or last point extrapolated values
	:rtype: float
	"""
	# extrapolation
	if time < df.index[0]:  # time before first in dataframe
		return df.iloc[0]  # return first row of (extrapolate using first/last element)
	elif time > df.index[len(df) - 1]:  # time later than last in dataframe
		return df.iloc[len(df) - 1]  # return last row (extrapolate using first/last element)
	else:
		# linear interpolation only for float type
		if df.dtype == float:
			ix_row1 = df.index.get_loc(time, method='ffill')
			time_row1 = df.index[ix_row1]  # time of closest row before time
			if time == time_row1:  # exact match of times
				row1 = df.iloc[ix_row1].copy()  # only copy is allowed to be returned
				return row1
			ix_row2 = df.index.get_loc(time, method='bfill')
			time_row2 = df.index[ix_row2]  # time of closest row after time
			row1 = df.iloc[ix_row1].copy()
			row2 = df.iloc[ix_row2].copy()
			return row1 + (time - time_row1) * (row2 - row1) / (time_row2 - time_row1)
		# take last value for all none float types
		else:
			ix_row = df.index.get_loc(time, method='nearest')
			row = df.iloc[ix_row].copy()
			return row


def get_pp_results(net, time):
	"""
	reads all result dataframes in network and stacks them to one row with current time as index
	:param net: pandapower network object
	:type net: pandapower.auxiliary.pandapowerNet
	:param time: current simulation time
	:type time: float
	:return: one row pandas dataframe including all results of pandapower network
	:rtype: pandas.DataFrame
	"""
	dfs = []
	df_dict = {key: getattr(net, key).copy() for key in net.keys() if key.startswith('res_') and not getattr(net,
																											 key).empty}
	if not df_dict:
		raise RuntimeError('No results present in pandapower network')
	# list of all copied (in order to leave actual pandapower results dataframes untouched) result dataframes in
	# network that are not empty
	keys = []
	for df_name, df in df_dict.items():
		# df.index = df.index.map(getattr(net, df_name[4:])['name'].to_dict()) # map index (bus IDs) to bus names using
		# information from net.bus dataframe
		df.index = df.index.to_series().map({key: str(val) for key, val in getattr(net, df_name[4:])[
			'name'].to_dict().items()})  # adds 'BUS: ' as identifier string to avoid loads with same names as bus and
		keys.append(df_name[4:])  # use for additional multiindex column level
		# sharing same variables, e.g., p_mw
		df = df.stack().to_frame().T  # stack dataframe to one row which later corresponds to the
		# current simulation time
		dfs.append(df)
	df_final = pd.concat(dfs,
						 axis=1,
						 keys=keys)  # combine all to a dataframe with one row (names of components should be unique!!)
	df_final.index = [time]  # index corresponds to current time
	return df_final
