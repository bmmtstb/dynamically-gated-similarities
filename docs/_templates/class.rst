.. Custom template for autosummary classes. Can be removed when
.. https://github.com/sphinx-doc/sphinx/issues/7912 is fixed.

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% if methods %}

   .. rubric:: {{ _('Methods') }}

   {% for item in methods %}

   .. automethod:: {{ name }}.{{ item }}

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
