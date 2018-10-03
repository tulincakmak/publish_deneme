


#önce bunları çalıştır
class BidGenerator:
   def __init__(self):
       self._bidgenerator = BidGeneratorHelper()

   def get_outputs(self):
       return [
           {
               "name": "partition_id",
               "type": "int"
           },
           {
               "name": "bid_amount",
               "type": "float"
           }
       ]

   def transform_frame(self, data_frame):
       """
       :rtype: pandas.DataFrame
       :returns: a data frame with partition_id and bid_amount columns
       :param data_frame: pandas.DataFrame input data
       :param element: ProjectElement current element
       """

       self._bidgenerator._validate(data_frame)
       values = data_frame.to_dict(orient="records")
       sorted_values = self._bidgenerator._sort_values_by_partition_id(values)
       result = list()
       for value in sorted_values:
           fixed_increment_value = str(value["fixed_increment_value"]) if str(
               value["fixed_increment_value"]) == "nan" or str(value["fixed_increment_value"]) == "Nan" or str(
               value["fixed_increment_value"]) == '' else float(value[
                                                                    "fixed_increment_value"])
           if not value["max_value"] >= value["base_value"]:
               raise ValueError(str(value["partition_id"]) + ": " + "Base value must be less than the Max value!")
           if not value["base_value"] >= value["min_value"]:
               raise ValueError(str(value["partition_id"]) + ": " + "Min value must be less than the Base value!")
           result = self._bidgenerator._exec_function(value["partition_id"], fixed_increment_value, value["min_value"],
                                                      value["base_value"],
                                                      value["max_value"], value["function"], value["number_of_values"],
                                                      result)
       return pd.DataFrame(result)



class BidGeneratorHelper:
    def _exec_function(self, partition_id, fixed_increment_value, min_value, base_value, max_value, function,
                       number_of_values, result):
        """

        :param partition_id:
        :param fixed_increment_value:
        :param min_value:
        :param base_value:
        :param max_value:
        :param function:
        :param number_of_values:
        :param result:
        :return:
        """
        functions = {
            "linear": self._linear,
            "normal": self._normal,
            "right_skewed": self._right_skewed,
            "left_skewed": self._left_skewed,
            "fixed_increment": self._fixed_increment
        }

        if fixed_increment_value not in ["nan", "NaN", "Nan", None, '', 0]:
            if base_value == min_value == max_value:
                result.extend(functions["fixed_increment"](fixed_increment_value, partition_id, min_value=0.01,
                                                           base_value=base_value))
            elif base_value == min_value != max_value:
                result.extend(functions["fixed_increment"](fixed_increment_value, partition_id, min_value=0.01,
                                                           base_value=base_value, max_value=max_value))
            elif base_value == max_value != min_value:
                result.extend(functions["fixed_increment"](fixed_increment_value, partition_id, min_value=min_value,
                                                           base_value=base_value, max_value=max_value))
            else:
                result.extend(
                    functions["fixed_increment"](fixed_increment_value, partition_id, min_value=min_value,
                                                 base_value=base_value, max_value=max_value))
        else:
            if base_value == min_value == max_value:
                result.extend(
                    functions[function](partition_id, number_of_values, min_value=0.01, base_value=base_value))
            elif base_value == min_value != max_value:
                result.extend(functions[function](partition_id, number_of_values, min_value=0.01,
                                                  base_value=base_value, max_value=max_value))
            elif base_value == max_value != min_value:
                result.extend(functions[function](partition_id, number_of_values, min_value=min_value,
                                                  base_value=base_value, max_value=max_value))
            else:
                result.extend(
                    functions[function](partition_id, number_of_values, min_value=min_value,
                                        base_value=base_value, max_value=max_value))

        return result

    def _validate(self, data_frame):
        data_frame = data_frame.set_index(['partition_id'])
        uniques = set(data_frame.index.values.tolist())
        if len(uniques) != data_frame.shape[0]:
            raise ValueError("The input must be revised, it possess duplicate partition ids!")

    def _sort_values_by_partition_id(self, values):
        """
        :param values: list of dicts
        :return: a sorted list of dicts of the values
        :rtype: list of dicts
        """

        return sorted(values, key=lambda item: item['partition_id'])

    def _fixed_increment(self, fixed_increment_value, partition_id, min_value=None, base_value=None, max_value=None):
        """
        :param min_value:
        :param max_value:
        :param fixed_increment_value:
        :param partition_id:
        :return: returns one partition id's bid values according to a fixed increment value
        :rtype: list of dicts
        """
        if max_value is None:
            left_part = base_value - min_value
            max_value = base_value + left_part
        bids = np.arange(min_value, max_value + fixed_increment_value / 2, fixed_increment_value)
        bids = [round(bid, 2) for bid in sorted(list(bids))]
        result = [{"partition_id": partition_id, "bid_amount": bid} for bid in bids]
        return result

    def _linear(self, partition_id, number_of_values, min_value=None, base_value=None, max_value=None):
        """
        :param min_value:
        :param base_value:
        :param max_value:
        :param number_of_values:
        :param partition_id:
        :return: return one partition id's bid values according to the linear distribution
        :rtype: list of dicts
        """
        if max_value is None:
            left_part = base_value - min_value
            max_value = base_value + left_part

        bids = np.linspace(min_value, max_value, number_of_values, endpoint=True)
        bids = [round(bid, 2) for bid in sorted(list(bids))]
        result = [{"partition_id": partition_id, "bid_amount": bid} for bid in bids]
        return result

    def _normal(self, partition_id, number_of_values, min_value=None, base_value=None, max_value=None):
        """
        :param min_value:
        :param base_value:
        :param max_value:
        :param number_of_values:
        :param partition_id:
        :return: returns one partition id's bid values according to the normal distribution
        :rtype: list of dicts
        """
        if max_value is None:
            left_part = base_value - min_value
            max_value = base_value + left_part
        sigma = (max_value - min_value)
        bids = truncnorm.rvs(
            (min_value - base_value) / sigma, (max_value - base_value) / sigma, loc=base_value, scale=sigma,
            size=number_of_values)
        bids = [round(bid, 2) for bid in sorted(list(bids))]
        result = [{"partition_id": partition_id, "bid_amount": bid} for bid in bids]
        return result

    def _right_skewed(self, partition_id, number_of_values, min_value=None, base_value=None, max_value=None):
        """
        :param min_value:
        :param base_value:
        :param max_value:
        :param number_of_values:
        :param partition_id:
        :return: returns one partition id's bid values according to the right skewed distribution
        :rtype: list of dicts
        """
        if max_value in None:
            left_part = base_value - min_value
            max_value = base_value + left_part
            right_part = max_value - base_value
            right_ratio = left_ratio = 1 / 2
        else:
            left_part = base_value - min_value
            right_part = max_value - base_value
            right_ratio = right_part / (max_value - min_value)
            left_ratio = left_part / (max_value - min_value)

        if right_part > left_part:
            right_number = int(number_of_values * right_ratio)
            left_number = int(number_of_values * left_ratio)
            while right_number + left_number != number_of_values:
                if right_number + left_number > number_of_values:
                    right_number -= 1
                elif right_number + left_number < number_of_values:
                    right_number += 1

        elif right_ratio == left_ratio:
            right_number = int(number_of_values * 2 / 3)
            left_number = int(number_of_values * 1 / 3)
            while right_number + left_number != number_of_values:
                if right_number + left_number > number_of_values:
                    right_number -= 1
                elif right_number + left_number < number_of_values:
                    right_number += 1

        else:
            right_number = int(number_of_values * left_ratio)
            left_number = int(number_of_values * right_ratio)
            while right_number + left_number != number_of_values:
                if right_number + left_number > number_of_values:
                    right_number -= 1
                else:
                    right_number += 1

        left_bids = np.linspace(min_value, base_value, left_number)
        right_bids = np.linspace(base_value, max_value, right_number)
        list(left_bids).extend(list(right_bids))
        bids = [round(bid, 2) for bid in sorted(list(left_bids))]
        result = [{"partition_id": partition_id, "bid_amount": bid} for bid in bids]
        return result

    def _left_skewed(self, partition_id, number_of_values, min_value=None, base_value=None, max_value=None):
        """
        :param min_value:
        :param base_value:
        :param max_value:
        :param number_of_values:
        :param partition_id:
        :return: returns one partition id's bid values according to the left skewed distribution
        :rtype: list of dicts
        """
        if max_value in None:
            left_part = base_value - min_value
            max_value = base_value + left_part
            right_part = max_value - base_value
            right_ratio = left_ratio = 1 / 2

        else:
            left_part = base_value - min_value
            right_part = max_value - base_value
            right_ratio = right_part / (max_value - min_value)
            left_ratio = left_part / (max_value - min_value)
        if right_part > left_part:
            right_number = int(number_of_values * left_ratio)
            left_number = int(number_of_values * right_ratio)
            while right_number + left_number != number_of_values:
                if right_number + left_number > number_of_values:
                    right_number -= 1
                else:
                    right_number += 1
        elif right_ratio == left_ratio:
            right_number = int(number_of_values * 1 / 3)
            left_number = int(number_of_values * 2 / 3)
            while right_number + left_number != number_of_values:
                if right_number + left_number > number_of_values:
                    right_number -= 1
                elif right_number + left_number < number_of_values:
                    right_number += 1

        else:
            right_number = int(number_of_values * right_ratio)
            left_number = int(number_of_values * left_ratio)
            while right_number + left_number != number_of_values:
                if right_number + left_number > number_of_values:
                    right_number -= 1
                else:
                    right_number += 1

        left_bids = np.linspace(min_value, base_value, left_number)
        right_bids = np.linspace(base_value, max_value, right_number)
        list(left_bids).extend(list(right_bids))
        bids = [round(bid, 2) for bid in sorted(list(left_bids))]
        result = [{"partition_id": partition_id, "bid_amount": bid} for bid in bids]
        return result







import pyodbc
import pandas as pd
import numpy as np



cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER=78.40.231.196;DATABASE=otelz__1521458750;UID=caglaS;PWD=c:agla12S')
cursor = cnxn.cursor()


#data=pd.resad_excel('bids11.xlsx')

sql="select * from ##bids"
data = pd.read_sql(sql,cnxn)

data = data.rename(columns={'trivagoId': 'partition_id'})

bid=BidGenerator()

bidgen = bid.transform_frame(data)



bidgen.to_excel('bid21.xlsx')















