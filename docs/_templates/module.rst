.. Custom template for autosummary modules. Can be removed when
.. https://github.com/sphinx-doc/sphinx/issues/7912 is fixed.

{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

   {% block attributes %}
       {% if attributes %}
           .. rubric:: {{ _('Module Attributes') }}

           .. autosummary::
              :toctree:
              :template: attribute.rst
           {% for item in attributes %}
              {{ item }}
           {%- endfor %}
       {% endif %}
   {% endblock %}

   {% block functions %}
       {% if functions %}
           .. rubric:: {{ _('Module Functions') }}

           .. autosummary::
              :toctree:
              :template: function.rst
           {% for item in functions %}
              {{ item }}
           {%- endfor %}
       {% endif %}
   {% endblock %}

   {% block classes %}
       {% if classes %}
           .. rubric:: {{ _('Module Classes') }}

           .. autosummary::
              :toctree:
              :template: class.rst
           {% for item in classes %}
              {{ item }}
           {%- endfor %}
       {% endif %}
   {% endblock %}

   {% block exceptions %}
       {% if exceptions %}
           .. rubric:: {{ _('Module Exceptions') }}

           .. autosummary::
              :toctree:
              :template: exception.rst
           {% for item in exceptions %}
              {{ item }}
           {%- endfor %}
       {% endif %}
   {% endblock %}

{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autosummary::
   :template: module.rst
   :toctree:
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
