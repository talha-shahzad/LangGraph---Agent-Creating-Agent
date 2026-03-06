import ast
import json
import logging
from typing import Any, Dict, Optional

class SafeExecutor:
    """
    Safely executes generated Python code using AST validation and a restricted namespace.
    """
    
    ALLOWED_MODULES = {
        'json', 'datetime', 'math', 're', 'random', 'requests', 'pytz', 'base64', 'hashlib',
        'uuid', 'time', 'collections', 'itertools', 'statistics', 'decimal'
    }
    
    ALLOWED_BUILTINS = {
        'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes', 'callable', 
        'chr', 'complex', 'divmod', 'enumerate', 'filter', 'float', 'format', 'frozenset', 
        'getattr', 'hasattr', 'hash', 'hex', 'id', 'int', 'isinstance', 'issubclass', 
        'iter', 'len', 'list', 'map', 'max', 'min', 'next', 'object', 'oct', 'ord', 
        'pow', 'print', 'property', 'range', 'repr', 'reversed', 'round', 'set', 
        'setattr', 'slice', 'sorted', 'str', 'sum', 'tuple', 'type', 'vars', 'zip',
        'dict', 'Exception', 'ValueError', 'TypeError', 'RuntimeError', 'StopIteration'
    }

    @staticmethod
    def _clean_code(code: str) -> str:
        """
        Cleans generated code by removing markdown blocks and fixing common syntax issues.
        """
        import re
        # Remove markdown code blocks if present
        code = re.sub(r'```python\s*(.*?)\s*```', r'\1', code, flags=re.DOTALL)
        code = re.sub(r'```\s*(.*?)\s*```', r'\1', code, flags=re.DOTALL)
        
        # Strip leading/trailing whitespace
        code = code.strip()
        
        # Fix "unexpected character after line continuation character"
        # This happens when there's a space/tab after a \
        code = re.sub(r'\\\s+\n', '\\\n', code)
        
        return code

    @staticmethod
    def validate_code(code: str) -> bool:
        """
        Performs AST analysis to ensure no dangerous operations are present.
        """
        try:
            # Clean before parsing
            clean_code = SafeExecutor._clean_code(code)
            tree = ast.parse(clean_code)
            for node in ast.walk(tree):
                # Block imports that aren't specifically allowed
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    module_name = ""
                    if isinstance(node, ast.Import):
                        module_name = node.names[0].name
                    else:
                        module_name = node.module or ""
                    
                    if module_name.split('.')[0] not in SafeExecutor.ALLOWED_MODULES:
                        logging.warning(f"Forbidden import attempt: {module_name}")
                        return False
                
                # Block attribute access to underscore properties
                if isinstance(node, ast.Attribute):
                    if node.attr.startswith('_'):
                        logging.warning(f"Forbidden private attribute access: {node.attr}")
                        return False
                
                # Allow classes and async functions, but still monitor dangerous nodes
                # Block potentially dangerous things like global/nonlocal in restricted ways if needed
                # For now, we allow ClassDef as requested by "less vigorous"
                pass
                    
            return True
        except Exception as e:
            logging.error(f"AST validation error: {str(e)}")
            # Log the code that failed validation for easier debugging
            logging.debug(f"Failed code snippet:\n{code[:500]}...")
            return False

    def execute(self, code: str, func_name: str, params: Dict[str, Any]) -> str:
        """
        Executes a function from the given code within a restricted namespace.
        """
        # Clean the code before validation and execution
        code = self._clean_code(code)
        
        if not self.validate_code(code):
            return "Error: Code validation failed. Dangerous operations detected or syntax error."

        # Prepare restricted globals with mapped builtins
        safe_builtins = {k: __builtins__[k] for k in self.ALLOWED_BUILTINS if k in __builtins__}
        
        # Add basic import functionality for allowed modules
        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in self.ALLOWED_MODULES:
                return __import__(name, globals, locals, fromlist, level)
            raise ImportError(f"Module '{name}' is not allowed")

        safe_builtins['__import__'] = safe_import
        
        restricted_globals = {
            '__builtins__': safe_builtins,
        }
        
        # Add allowed modules to namespace
        import importlib
        for module in self.ALLOWED_MODULES:
            try:
                restricted_globals[module] = importlib.import_module(module)
            except ImportError:
                continue

        try:
            # Execute the code to define the function in our restricted namespace
            tree = ast.parse(code)
            exec(compile(tree, filename="<string>", mode="exec"), restricted_globals)
            
            # Auto-detect function name if the provided one is missing
            if func_name not in restricted_globals:
                # Find all top-level functions defined in the code
                functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
                if functions:
                    # Use the first function found as a fallback
                    func_name = functions[0]
                    logging.info(f"Auto-detected function name: {func_name}")
                else:
                    # Maybe it's a class?
                    classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
                    if classes:
                        func_name = classes[0]
                        logging.info(f"Auto-detected class/constructor name: {func_name}")
                    else:
                        return f"Error: No function or class definition found in generated code."
            
            if func_name not in restricted_globals:
                return f"Error: Function/Class '{func_name}' not found in generated code."
            
            obj = restricted_globals[func_name]
            
            # Execute the function or instantiate the class
            if callable(obj):
                result = obj(**params)
                return str(result)
            else:
                return f"Error: '{func_name}' is not callable."
            
        except Exception as e:
            logging.exception(f"Execution error in tool {func_name}")
            return f"Error during execution: {str(e)}"
