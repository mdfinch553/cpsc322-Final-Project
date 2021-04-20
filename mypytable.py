# TODO: copy your mypytable.py solution from PA2 here
import copy
import csv 
from tabulate import tabulate
# uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data. There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            tuple of int: rows, cols in the table

        Notes:
            Raise ValueError on invalid col_identifier
        """
        col_index = self.column_names.index(col_identifier) 
        col = [] 
        for row in self.data: 
            if (row[col_index] != "NA"):
                col.append(row[col_index])
            elif (include_missing_values == True):
                col.append(row[col_index])
        return col

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        numeric_data = []
        for row in self.data: 
            numeric_row = []
            for item in row: 
                try:
                    float_val = float(item)
                    numeric_row.append(float_val)
                except: 
                    numeric_row.append(item)
                    continue
            numeric_data.append(numeric_row)
        self.data = numeric_data

    def drop_rows(self, rows_to_drop):
        """Remove rows from the table data.

        Args:
            rows_to_drop(list of list of obj): list of rows to remove from the table data.
        """
        drop_table = copy.deepcopy(self.data)
        for row in self.data: 
            if row in rows_to_drop: 
                drop_table.remove(row) 
                rows_to_drop.remove(row)
        self.data = drop_table

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like: table = MyPyTable().load_from_file(fname)
        
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        table = []
        with open(filename, "r") as f:
            reader = csv.reader(f)
            lines = list(reader)
            table = lines
        self.column_names = table[0]
        self.data = table
        self.data.remove(self.column_names)
        self.convert_to_numeric()
        return self 

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.column_names)
            for row in self.data: 
                writer.writerow(row)

    def find_duplicates(self, key_column_names):
        """Returns a list of duplicates. Rows are identified uniquely baed on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns: 
            list of list of obj: list of duplicate rows found
        
        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s). The first instance is not
                considered a duplicate.
        """
        duplicates = []
        indexes = [] 
        rows = []
        for name in key_column_names: 
            indexes.append(self.column_names.index(name))
        for row in self.data: 
            temp_row = []
            for index in indexes: 
                temp_row.append(row[index])
            if temp_row in rows: 
                duplicates.append(row)
            else: 
                rows.append(temp_row)
        return duplicates 

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        index = 0
        new_table = copy.deepcopy(self.data)
        rowsToRemove = []
        for title in self.column_names:  
            index = self.column_names.index(title)
    
            for row in new_table: 
                if row[index] == 'NA' or row[index] == "N/A" or row[index] == '':  
                    if row not in rowsToRemove:
                        rowsToRemove.append(row)
    
        for row in rowsToRemove: 
            new_table.remove(row)

        self.data = new_table 

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        index = self.column_names.index(col_name)
        new_table = copy.deepcopy(self.data) 
        self.convert_to_numeric()
         
        avgs = []
        replace_indexes = []
        avg = 0       
        for row in new_table: 
            if row[index] == "NA" or row[index] == '' or row[index] == "N/A": 
                replace_indexes.append(new_table.index(row))
            elif type(row[index]) == float or type(row[index]) == int: 
                avgs.append(row[index])
                avg = sum(avgs) / len(avgs) 
        for i in replace_indexes: 
            new_table[i][index] = avg
            
        self.data = new_table 

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed.
        """
       
        self.convert_to_numeric()
        table = []
        temp_data = copy.deepcopy(self.data)
        for name in col_names: 
            stats = []
            stats.append(name)
            col = self.get_column(name)
            temp_col = copy.deepcopy(col)
            for item in col: 
                if item == "NA" or item == "": 
                    temp_col.remove(item)
            col = temp_col
            if col: 
                stats.append(min(col))
                stats.append(max(col))
                stats.append((max(col) + min(col))/2)
                stats.append(sum(col)/len(col))
                col.sort()
                if len(col) % 2 == 0: 
                    m1 = col[len(col)//2] 
                    m2 = col[len(col)//2 - 1] 
                    median = (m1 + m2)/2
                else: 
                    median = col[len(col)//2] 
                stats.append(median)
                table.append(stats)
        return MyPyTable(col_names, table)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        joined_table = []
        joined_header = self.column_names.copy()
        for col_name in other_table.column_names:
            if col_name not in joined_header:
                joined_header.append(col_name)
        for row in self.data:
            for i in range(len(other_table.data)):
                match = True 
                for col_name in key_column_names:
                    index = self.column_names.index(col_name)
                    if row[index] not in other_table.data[i]:
                        match = False 
                        break
                    if match == True:
                        temp_row = row 
                        for j in range(len(other_table.data[i])):
                            if other_table.column_names[j] not in self.column_names:
                                temp_row.append(other_table.data[i][j])
                        joined_table.append(temp_row)
        return MyPyTable(joined_header, joined_table)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        table = self.perform_inner_join(other_table, key_column_names) 
        table_data = copy.deepcopy(table.data) 
        indexes1 = []
        indexes2 = []
        indexes3 = []
        rows = []
        other_rows = []
        data_rows = []
        for name in key_column_names: 
            indexes3.append(table.column_names.index(name))
            indexes1.append(self.column_names.index(name))
            indexes2.append(other_table.column_names.index(name))
        for row in table_data: 
            temp_row = []
            for index in indexes3: 
                value = copy.deepcopy(row[index])
                temp_row.append(value)
            rows.append(temp_row)
        for row in self.data: 
            temp_row = []
            for index in indexes1: 
                value = copy.deepcopy(row[index])
                temp_row.append(value)
            if temp_row not in rows: 
                data_rows.append(row)  
        for row in other_table.data: 
            temp_row = []
            for index in indexes2: 
                value = copy.deepcopy(row[index])
                temp_row.append(value)
            if temp_row not in rows: 
                other_rows.append(row)  
        for row in data_rows: 
            for name in table.column_names:
                if name not in self.column_names: 
                    if table.column_names.index(name) > len(row) - 1: 
                        row.append("NA") 
                    else: 
                        row.insert(table.column_names.index(name), "NA")
            table.data.append(row)
        for row in other_rows: 
            for name in table.column_names:
                if name not in other_table.column_names: 
                        if table.column_names.index(name) > len(row) - 1: 
                            row.append("NA") 
                        else: 
                            row.insert(table.column_names.index(name), "NA")
            table.data.append(row)
            
           

        return table