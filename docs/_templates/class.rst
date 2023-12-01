.. Custom template for autosummary classes. Can be removed when
.. https://github.com/sphinx-doc/sphinx/issues/7912 is fixed.

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
       .. automethod:: __init__

       {% if methods %}
           .. rubric:: {{ _('Methods') }}

           .. autosummary::
              :template: method.rst
              :toctree:
           {% for item in methods %}
              {% if '__init__' != item %}
                ~{{ name }}.{{ item }}
             {% endif %}
           {%- endfor %}
       {% endif %}
   {% endblock %}

   {% block attributes %}
       {% if attributes %}
           .. rubric:: {{ _('Attributes') }}

           .. autosummary::
              :template: attribute.rst
              :toctree:
           {% for item in attributes %}
              ~{{ name }}.{{ item }}
           {%- endfor %}
       {% endif %}
   {% endblock %}
